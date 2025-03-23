import tensorflow as tf

"""
https://github.com/zhezh/focalloss/blob/master/focalloss.py
https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/

The focusing parameter γ(gamma) smoothly adjusts the rate at which easy examples are down-weighted. 
When γ = 0, focal loss is equivalent to categorical cross-entropy, and as γ is increased the effect 
of the modulating factor is likewise increased (γ = 2 works best in experiments).
α(alpha): balances focal loss, yields slightly improved accuracy over the non-α-balanced form.  
"""
@tf.keras.saving.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    # y_true = tf.to_int64(y_true)
    # y_true = tf.convert_to_tensor(y_true, tf.int64)
    # y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    # num_cls = y_pred.shape[1]

    model_out = tf.math.add(y_pred, epsilon)
    # onehot_labels = tf.math.one_hot(labels, num_cls)
    ce = tf.math.multiply(y_true, -tf.math.log(model_out))
    weight = tf.math.multiply(y_true, tf.math.pow(tf.math.subtract(1., model_out), gamma))
    fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
    reduced_fl = tf.math.reduce_max(fl, axis=1)
    # reduced_fl = tf.math.reduce_sum(fl, axis=1)  # same as reduce_max
    return tf.math.reduce_mean(reduced_fl)