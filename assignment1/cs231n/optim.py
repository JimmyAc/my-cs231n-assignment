import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # v_t = \text{momentum} \cdot v_{t-1} - \text{learning_rate} \cdot dw  
    # w_{t} = w_{t-1} + v_t  
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # \text{cache}_t = \text{decay_rate} \cdot \text{cache}_{t-1} + (1 - \text{decay_rate}) \cdot (dw^2)
    # w_t = w_{t-1} - \text{learning_rate} \cdot \frac{dw}{\sqrt{\text{cache}_t + \epsilon}}
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dw ** 2)
    next_w = w - config['learning_rate'] * dw / ((config['cache']**0.5) + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################

    # 更新迭代次数
    # t = t + 1
    # 更新一阶矩（梯度的移动平均，类似动量）
    # m_t = \text{beta1} \cdot m_{t-1} + (1 - \text{beta1}) \cdot dw \quad \text{（一阶矩：梯度均值）}
    # m = beta1 * m + ( 1 - beta1 ) * dw
    # 更新二阶矩（梯度平方的移动平均，类似RMSProp的cache）
    # v_t = \text{beta2} \cdot v_{t-1} + (1 - \text{beta2}) \cdot (dw^2) \quad \text{（二阶矩：梯度平方均值）}
    # v = beta2 * v + ( 1 - beta2 ) * dW^2 
    # 计算偏差修正后的一阶矩和二阶矩
    # \hat{m}_t = \frac{m_t}{1 - \text{beta1}^t} \quad \text{（偏差修正后的梯度均值）}
    # \hat{v}_t = \frac{v_t}{1 - \text{beta2}^t} \quad \text{（偏差修正后的梯度平方均值）}
    # m_hat = m / ( 1 - beta1^t )
    # v_hat = v / ( 1 - beta2^t )
    # 更新权重
    # w_t = w_{t-1} - \text{learning_rate} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    
    config["t"] += 1
    
    # 更新一阶矩（梯度的移动平均，类似动量）
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    
    # 更新二阶矩（梯度平方的移动平均，类似RMSProp的cache）
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dw **2)
    
    # 计算偏差修正后的一阶矩和二阶矩
    m_hat = config["m"] / (1 - config["beta1"]** config["t"])
    v_hat = config["v"] / (1 - config["beta2"] ** config["t"])
    
    # 更新权重
    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
