import tensorflow as tf
import numpy as np


class BertLoss(tf.keras.layers.Layer):
    """设置mlm预训练任务的loss。
    """

    def __init__(self, **kwargs):
        super(BertLoss, self).__init__(**kwargs)

    def call(self, inputs):
        mlm_pred, nsp_pred, origin_x, batch_mlm_mask, batch_y = inputs
        mlm_loss = tf.keras.losses.sparse_categorical_crossentropy(origin_x, mlm_pred)
        mlm_loss = mlm_loss * batch_mlm_mask
        mlm_loss = tf.reduce_sum(mlm_loss) / (tf.reduce_sum(batch_mlm_mask) + 1)

        nsp_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, nsp_pred)
        nsp_loss = tf.reduce_mean(nsp_loss)

        mask_position = np.argwhere(batch_mlm_mask.numpy())
        mask_predict = tf.gather_nd(tf.math.argmax(mlm_pred, axis=-1, output_type=tf.int32), mask_position)
        mask_real = tf.gather_nd(origin_x, mask_position)
        correct_num = tf.reduce_sum(tf.cast(mask_predict == mask_real, dtype=tf.int32))
        mlm_acc = correct_num / mask_position.shape[0]

        nsp_predict = tf.math.argmax(nsp_pred, axis=-1, output_type=tf.int32)
        correct_nsp_num = tf.reduce_sum(tf.cast(nsp_predict == mask_real, dtype=tf.int32))
        nsp_acc = correct_nsp_num / len(batch_y)

        return mlm_loss, nsp_loss, mlm_acc, nsp_acc
