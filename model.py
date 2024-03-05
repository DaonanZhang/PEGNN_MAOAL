import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from scipy import sparse
import torch.nn.parallel
import torch.utils.data
from spatial_utils import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))

    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2


def extract_first_element_per_batch(tensor1, tensor2):
    # Get the unique batch indices from tensor2
    unique_batch_indices = torch.unique(tensor2)
    # Initialize a list to store the first elements of each batch item
    first_elements = []

    # Iterate through each unique batch index
    for batch_idx in unique_batch_indices:
        # Find the first occurrence of the batch index in tensor2
        first_occurrence = torch.nonzero(tensor2 == batch_idx, as_tuple=False)[0, 0]
        # Extract the first element from tensor1 and append it to the list
        first_element = tensor1[first_occurrence]
        first_elements.append(first_element)
    # Convert the list to a tensor
    result = torch.stack(first_elements, dim=0)
    return result


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    # Batch_norm: same dimension, different features, different examples
    # layer_norm: same features, same examples, different dim

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # output shape is the same as x
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=0.5,
                 hidden_dim=-1,
                 activation="relu",
                 use_layernormalize=True,
                 skip_connection=False,
                 context_str=None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))



    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim

        output = input_tensor
        for layer in self.layers:
            # applied in each layer
            output = layer(output)

        return output


class SpatialEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding
    """

    def __init__(self, spa_embed_dim, coord_dim=2, settings=None, ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space
        """
        super(SpatialEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.ffn = ffn
        # input_dim:2
        self.input_embed_dim = self.coord_dim
        self.nn_length = settings['nn_length']
        self.nn_hidden_dim = settings['nn_hidden_dim']
        if self.ffn is not None:
            # by creating the ffn, the weights are initialized use kaiming_init
            self.ffn = MultiLayerFeedForwardNN(self.input_embed_dim, spa_embed_dim,
                                               num_hidden_layers=settings['nn_length'],
                                               hidden_dim=settings['nn_hidden_dim'],
                                               dropout_rate=settings['dropout_rate'])


    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = coords
        # Feed Forward Network
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

class PEGCN(nn.Module):

    """
        GCN with positional encoder and auxiliary tasks
    """

    # default parameters
    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k=20, conv_dim=64,
                 aux_task_num = 0, settings=None):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features_in = num_features_in
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k
        self.nn_length = settings['nn_length']
        self.nn_hidden_dim = settings['nn_hidden_dim']
        self.aux_task_num = settings['aux_task_num']

        self.spenc = SpatialEncoder(
            spa_embed_dim=emb_hidden_dim, ffn=True, settings=settings
        )

        # decrease the dimension of the embedding
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )

        # feature dim:12
        # self.inc_transformer = nn.Linear(12, settings['d_model'])
        # 13 = 12 features for rest features and 1 for Q
        self.inc = nn.Linear(in_features=12, out_features=settings['d_model'])
        encoder_layers = TransformerEncoderLayer(settings['d_model'], settings['nhead'],
                                                      settings['dim_feedforward'], settings['transformer_dropout'], batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, settings['num_encoder_layers'])

        # x_l + emb_l + q_l
        self.conv1 = GCNConv(num_features_in + emb_dim + 1, conv_dim)
        self.conv2 = GCNConv(conv_dim, conv_dim)

        # # use ffc as task heads
        # self.fc = nn.Linear(conv_dim, num_features_out)

        self.task_heads = nn.ModuleList()

        for i in range(0, aux_task_num + 1):
            head = MultiLayerFeedForwardNN(input_dim=conv_dim, output_dim=num_features_out,
                                    num_hidden_layers=settings['heads']['nn_length'],
                                    hidden_dim=settings['heads']['nn_hidden_dim'],
                                    dropout_rate=settings['heads']['dropout_rate'],)
            self.task_heads.append(head)
        #     already use kaiming_init by creating

        # self.noise_sigma = torch.nn.Parameter(torch.tensor([0.1, ], device=self.device))


        self.Q = None
        # MTH

        # init weights
        for p in self.dec.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv1.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv2.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        # torch.nn.init.kaiming_normal_(self.fc.weight)
        # torch.nn.init.kaiming_normal_(self.task_heads[0].weight)

    def forward(self, inputs, targets, aux_answers, coords, input_lengths, rest_features, head_only):

        # inputs.shape x: torch.Size([32, 43, 2])
        # 2: (value, rank)

        # input_lenths.shape: torch.Size([32])
        # batch_size

        # rest_features.shape = torch.Size([32, 187, 12])
        # self.Q.shape = torch.Size([32, 187, 1])
        # rest_feature_Q.shape = torch.Size([32, 187, 13])

        # ________________________________________postional encoding_______________________________________________________
        emb = self.spenc(coords)
        # decrease the dimension of the embedding to the emb_dim
        emb = self.dec(emb)
        # emb.shape: torch.Size([32, 47, 16])
        emb_l, indexer = padded_seq_to_vectors(emb, input_lengths)
        # emb_l.shape after padding: torch.Size([1376, 16])

        # _________________________________________env_features_______________________________________________________

        self.Q = torch.nn.Parameter(torch.ones(rest_features.shape[0], rest_features.shape[1], 1, device=self.device))

        rest_feature_Q = torch.cat([self.Q, rest_features], dim=2)
        # rest_feature_Q.shape after concat: torch.Size([8, 175, 12])

        rest_feature_Q = self.inc(rest_feature_Q)

        feature_emb = self.transformer_encoder(rest_feature_Q)

        Q_feature_emb = feature_emb[:,:, 0]
        Q_feature_emb = Q_feature_emb.unsqueeze(-1)
        # Q_feature_emb.shape = torch.Size([32, 212, 1])

        q_l, _ = padded_seq_to_vectors(Q_feature_emb, input_lengths)
        # rest_featrues_l.shapetorch.Size([1509, 1])

        # _________________________________________features_______________________________________________________
        x_l, _ = padded_seq_to_vectors(inputs, input_lengths)
        # x_l_shape = torch.Size([1376, 2])

        if self.num_features_in == 2:
            first_element = x_l[:, 0].unsqueeze(-1)
            last_element = x_l[:, -1].unsqueeze(-1)
            x_l = torch.cat([first_element, last_element], dim=-1)

        edge_index = knn_graph(emb_l, k=self.k, batch=indexer)
        # edge_index.shape = torch.size([2, 29840])
        edge_weight = makeEdgeWeight(emb_l, edge_index).to(self.device)
        # edge_weight.shape: torch.Size([27520])

        # concat the embedding with the input
        x = torch.cat((x_l, emb_l, q_l), dim=1)
        # => 19 = 2 x_l + 16 pe + 1 q_l
        # x.shape_ after concat: torch.Size([1509, 19])

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.task_heads[0](h2)

        y_l, _ = padded_seq_to_vectors(targets, input_lengths)
        # y_l.shape after padding: torch.Size([1376, 1])

        # _____________________________________Aux_answers_____________________________________
        # length = 3
        aux_y_ls = []
        for i in range(0, self.aux_task_num):
            aux_y_l, _ = padded_seq_to_vectors(aux_answers[i], input_lengths)
            aux_y_ls.append(aux_y_l)

            # aux_y_l.shape: torch.Size([398, 1])

        # _____________________________________Aux_outputs_____________________________________
        # length = 3
        aux_outputs = []

        # task_head[0] only for primary task
        for i in range(1, self.aux_task_num + 1):
            aux_output = self.task_heads[i](h2)
            aux_outputs.append(aux_output)


        # the result of output and y_l, in order to compare the BMC bzw. lost function
        # return loss now
        if not head_only:
            # debug mode not calculate loss in model
            return output, y_l, aux_outputs, aux_y_ls
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            if len(self.task_heads) == 1:
                aux_output_head = []
                aux_target_head = []
            else:
                aux_output_head = [extract_first_element_per_batch(aux_output, indexer) for aux_output in aux_outputs]
                aux_target_head = [extract_first_element_per_batch(aux_y_l, indexer) for aux_y_l in aux_y_ls]


            # print('output_head.shape:', output_head.shape)
            # print('target_head.shape:', target_head.shape)
            # print('aux_output_head.shape:', aux_output_head[0].shape)
            # print('aux_target_head.shape:', aux_target_head[0].shape)

            return output_head, target_head, aux_output_head, aux_target_head
    #
    # #     calculate the loss
    # def bmc_loss(self, pred, target):
    #     """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    #     Args:
    #       pred: A float tensor of size [batch, 1].
    #       target: A float tensor of size [batch, 1].
    #       noise_var: A float number or tensor.
    #     Returns:
    #       loss: A float tensor. Balanced MSE Loss.
    #     """
    #
    #     noise_var = self.noise_sigma ** 2
    #     batch_size = pred.shape[0]
    #
    #     # MASK VALUE = -1 in Dataloader => -1 means doesn't exist
    #     mask = (target != -1).float().squeeze()
    #     logits = - (pred - target.T).pow(2) / (2 * noise_var)
    #     logits_masked = logits * mask
    #
    #     loss = F.cross_entropy(logits_masked, torch.arange(batch_size).to(self.device))  # contrastive-like loss
    #     loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable
    #     return loss
