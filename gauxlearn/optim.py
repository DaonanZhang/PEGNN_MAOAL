from torch.nn.utils.clip_grad import clip_grad_norm_

from gauxlearn.implicit_diff import Hypergrad


class MetaOptimizer:

    def __init__(self, meta_optimizer, hpo_lr, truncate_iter=3, max_grad_norm=10):
        """Auxiliary parameters optimizer wrapper

        :param meta_optimizer: optimizer for auxiliary parameters
        :param hpo_lr: learning rate to scale the terms in the Neumann series
        :param truncate_iter: number of terms in the Neumann series
        :param max_grad_norm: max norm for grad clipping
        """
        self.meta_optimizer = meta_optimizer
        self.hypergrad = Hypergrad(learning_rate=hpo_lr, truncate_iter=truncate_iter)
        self.max_grad_norm = max_grad_norm

    def step(self, train_grads, val_loss,shared_parameters, aux_params, return_grads=False):
        """

        :param train_loss: train loader
        :param val_loss:
        :param parameters: parameters (main net)
        :param aux_params: auxiliary parameters
        :param return_grads: whether to return gradients
        :return:
        """
        # zero grad
        self.zero_grad()

        # validation loss
        hyper_gards = self.hypergrad.grad(
            loss_val=val_loss,
            grad_train=train_grads,
            aux_params=aux_params,
            shared_params=shared_parameters
        )

        # pseudocode: line 22
        for p, g in zip(aux_params, hyper_gards):
            p.grad = g

        # grad clipping makes the normlization

# aux_params：这是包含模型参数的列表或张量。通常，这是模型的权重和偏置参数。
# max_norm：这是梯度的最大范数（norm）的阈值。梯度的范数是梯度向量的长度。
# clip_grad_norm_ 函数的作用是计算模型的梯度向量的范数，并将其裁剪到不超过 max_norm 的大小。如果梯度的范数大于 max_norm，则梯度向量会被重新缩放，使其不超过这个阈值，以防止梯度爆炸的发生。
# 梯度剪裁有助于提高深度学习模型的稳定性，减少训练过程中的梯度问题，并更容易使模型收敛到合适的解。
# 这对于训练深层神经网络非常有用，特别是在使用循环神经网络（RNN）等容易受到梯度爆炸问题影响的模型时。通过设置合适的 max_grad_norm 阈值，可以控制梯度剪裁的程度。
        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)

        # meta step
        # MAOAL code: line 23
        self.meta_optimizer.step()
        if return_grads:
            return hyper_gards

    def zero_grad(self):
        self.meta_optimizer.zero_grad()
