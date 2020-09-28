import tensorflow as tf


class MLMLoss(tf.keras.layers.Layer):
    """设置mlm预训练任务的loss。
    mlm任务的输出为模型输出接一个dense然后再过模型中token embedding的转置，变回为word的onehot格式
    loss就是这个onehot和原本的onehot格式计算多分类交叉熵"""
    def __init__(self, **kwargs):
        super(MLMLoss, self).__init__(**kwargs)

    def call(self, inputs):
        y_true, y_pred, mlm_mask = inputs
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        loss = loss * mlm_mask
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(mlm_mask) + 1)
        return loss
