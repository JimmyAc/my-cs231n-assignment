"""This file defines layer types that are commonly used for recurrent neural networks.
"""
import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    """
    next_h = None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN.                 #
    ##############################################################################
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    """
    h = None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # 获取维度信息
    N, T, D = x.shape
    H = h0.shape[1]

    # 初始化输出张量 h，形状为 (N, T, H)，用于存储每个时间步的隐藏状态
    # 保持与输入 x 相同的数据类型和设备（CPU/GPU）
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)

    # 初始化“上一个隐藏状态”为 h0
    prev_h = h0

    # 循环遍历每一个时间步 (0 到 T-1)
    for t in range(T):
        # 1. 取出当前时刻的输入数据 xt
        # x 的形状是 (N, T, D)，取出第 t 个时刻，形状变为 (N, D)
        xt = x[:, t, :]

        # 2. 调用你之前写好的单步前向传播函数
        # 输入：当前数据 xt，上一步的隐藏状态 prev_h，以及权重参数
        # 输出：当前这一步计算出的新隐藏状态
        next_h = rnn_step_forward(xt, prev_h, Wx, Wh, b)

        # 3. 将计算结果存入输出张量 h 的对应位置
        h[:, t, :] = next_h

        # 4. 更新 prev_h，为下一次循环做准备
        # 当前计算出的 hidden state 将成为下一个时间步的 input hidden state
        prev_h = next_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    out = None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using Pytorch's array indexing.         #
    ##############################################################################
    out = W[x]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # 1. 计算总的激活值 (Activation)
    # 形状: (N, D)@(D, 4H) + (N, H)@(H, 4H) + (4H,) -> (N, 4H)
    activation = x @ Wx + prev_h @ Wh + b

    # 2. 将激活值切分为 4 个部分: input(i), forget(f), output(o), gate(g)
    # 每个部分的形状为 (N, H)
    # chunk(4, dim=1) 会将张量在第1维切分成4等份
    a_i, a_f, a_o, a_g = activation.chunk(4, dim=1)

    # 3. 应用非线性激活函数
    i = torch.sigmoid(a_i)  # Input gate
    f = torch.sigmoid(a_f)  # Forget gate
    o = torch.sigmoid(a_o)  # Output gate
    g = torch.tanh(a_g)     # Gate gate (Cell candidate)

    # 4. 更新细胞状态 (Cell State) c_t = f * c_{t-1} + i * g
    next_c = f * prev_c + i * g

    # 5. 更新隐藏状态 (Hidden State) h_t = o * tanh(c_t)
    next_h = o * torch.tanh(next_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # 获取维度信息
    N, T, D = x.shape
    H = h0.shape[1]

    # 初始化输出张量 h，形状为 (N, T, H)
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)

    # 初始化状态
    prev_h = h0
    # LSTM 需要初始的细胞状态 c0，通常初始化为全零
    prev_c = torch.zeros_like(h0)

    # 时间步循环
    for t in range(T):
        # 取出当前时刻的输入数据 xt: (N, D)
        xt = x[:, t, :]

        # 执行单步前向传播
        next_h, next_c = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)

        # 存储当前时刻的隐藏状态
        h[:, t, :] = next_h

        # 更新状态以供下一次迭代使用
        prev_h = next_h
        prev_c = next_c

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
