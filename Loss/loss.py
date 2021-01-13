import tensorflow as tf


class BERT_Loss(tf.keras.layers.Layer):

    def __init__(self):
        super(BERT_Loss, self).__init__()

    def call(self, inputs):
        (mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y) = inputs

        x_pred = tf.nn.softmax(mlm_predict, axis=-1)
        mlm_loss = tf.keras.losses.sparse_categorical_crossentropy(origin_x, x_pred)
        mlm_loss = tf.math.reduce_sum(mlm_loss * batch_mlm_mask, axis=-1) / (tf.math.reduce_sum(batch_mlm_mask, axis=-1) + 1)

        y_pred = tf.nn.softmax(nsp_predict, axis=-1)
        nsp_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, y_pred)

        return nsp_loss, mlm_loss



