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
import time


# no usage?
def length_to_mask(lengths, total_len, device):
    max_len = total_len
    # torch.arange(max_len) 创建了一个从 0 到 max_len-1 的一维张量。
    # .expand(lengths.shape[0], max_len) 扩展了这个一维张量，使其成为一个二维张量，其中的每一行都包含相同的整数序列。
    # < lengths.unsqueeze(1) 使用逐元素小于比较，将这个二维张量与经过扩展的 lengths 张量进行比较。lengths.unsqueeze(1)
    # 用于将 lengths 张量从一维扩展为二维，以便进行逐元素比较。
    # 结果是一个形状为 (batch_size, max_len) 的二维布尔张量 mask，其中的每个元素表示相应位置上的值是否小于 lengths 中对应位置的值。
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    # false->0, true->1
    mask = mask.int()
    return mask


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

        # yes, it's layer normalization

        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
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



# 这个函数的作用是根据用户指定的初始化方式（"random" 或 "geometric"）生成一组频率值，并将其存储在 freq_list 中
# 3.2 in the paper: PE(c,tau_min,tau_max, seta_pe) = NN(ST(c,tau_min,tau_max), seta_pe)?

# why in this form, not in the form of cos(C_v / (tau_min * g ** (s/(s-1)))) and sin (C_v / (tau_min * g ** (s/(s-1))))?
# for context-aware spatial coordinaate embedding?
def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    freq_list = None
    if freq_init == "random":
        freq_list = torch.rand(frequency_num) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        timescales = min_radius * torch.exp(torch.arange(frequency_num, dtype=torch.float32) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list


# ******************************* Encoder ***********************************
# enhance the spatial relation encoder from nn.module? but why only nn.module?
# Is this so inteligent? ..
class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric",
                 ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        # torch.unsqueeze(self.freq_list, 1) 是将self.freq_list的维度进行扩展的操作。
        # 具体地说，它将self.freq_list中的数据保持不变，并在第一个维度（维度索引为0）之前插入一个新的维度，该新维度的大小为1。
        freq_mat = torch.unsqueeze(self.freq_list, 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = freq_mat.repeat(1, 2)


    def make_input_embeds(self, coords):
        # coords: shape (batch_size, num_context_pt, 2)
        batch_size, num_context_pt, _ = coords.shape
        # coords: shape (batch_size, num_context_pt, 2, 1, 1)
        coords = coords.unsqueeze(-1).unsqueeze(-1)
        # coords: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords = coords.repeat(1, 1, 1, self.frequency_num, 2)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords * self.freq_mat.to(self.device)
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = torch.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = torch.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = torch.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):

        # embedding function

        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # Feed Forward Network
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class PEGCN(nn.Module):

    # Hole right part of PEGCN consider the figure 1 in the paper: GCNCONV layers?

    """
        GCN with positional encoder and auxiliary tasks
    """

    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k=20, conv_dim=64):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features_in = num_features_in
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k

        # an instance of the encoder:GridcellsSpatialRelationEncoder
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=emb_hidden_dim, ffn=True, min_radius=1e-06, max_radius=360
        )

        # decrease the dimension of the embedding
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )


        self.conv1 = GCNConv(num_features_in + emb_dim, conv_dim)
        self.conv2 = GCNConv(conv_dim, conv_dim)
        # fully connected layer
        self.fc = nn.Linear(conv_dim, num_features_out)
        
        self.noise_sigma = torch.nn.Parameter(torch.tensor([0.1, ], device=self.device))
        
        # init weights
        for p in self.dec.parameters():
            if p.dim() > 1:
                # torch.nn.init.kaiming_normal_ 使用Kaiming初始化方法，
                # 该方法是一种针对ReLU（修正线性单元）激活函数设计的初始化策略。
                # 它的目标是使权重初始化在训练初始阶段产生的梯度方差接近于1，以防止梯度爆炸或梯度消失问题。
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv1.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv2.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, inputs, targets, coords, input_lenths, head_only):
        # start_time = time.time()
        # encoding
        emb = self.spenc(coords)
        # spenc_time = time.time()
        # decrease the dimension of the embedding to the emb_dim
        emb = self.dec(emb)
        # dec_time = time.time()

        # why should there a transformation between padded_seq_to_vectors?
        emb_l, indexer = padded_seq_to_vectors(emb, input_lenths)
        x_l, _ = padded_seq_to_vectors(inputs, input_lenths)
        if self.num_features_in == 2:
            first_element = x_l[:, 0].unsqueeze(-1)
            last_element = x_l[:, -1].unsqueeze(-1)
            x_l = torch.cat([first_element, last_element], dim=-1)
        y_l, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _ = padded_seq_to_vectors(coords, input_lenths)


        # ptv_time = time.time()
        
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)
        edge_weight = makeEdgeWeight(c_l, edge_index).to(self.device)
        # edge_time = time.time()

        # concat the embedding with the input
        x = torch.cat((x_l, emb_l), dim=1)
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        # CNN_time = time.time()
        
        # print(f'spenc: {spenc_time-start_time}, dec: {dec_time-spenc_time}, ptv: {ptv_time-dec_time}, edge: {edge_time-ptv_time}, CNN: {CNN_time-edge_time}')


        # the result of output and y_l, in order to compare the BMC bzw. lost function
        if not head_only:
            return output, y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return output_head, target_head

    #     calculate the loss
    def bmc_loss(self, pred, target):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, 1].
          target: A float tensor of size [batch, 1].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        noise_var = self.noise_sigma ** 2
        logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(self.device))     # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
        return loss
