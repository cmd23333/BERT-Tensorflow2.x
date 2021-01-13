import tensorflow as tf


def calculate_pretrain_task_accuracy(nsp_predict, mlm_predict, batch_mlm_mask, origin_x, batch_y):
    y_predict = tf.math.argmax(nsp_predict, axis=-1)
    nsp_accuracy = tf.keras.metrics.Accuracy()
    nsp_accuracy.update_state(y_predict, batch_y)
    nsp_accuracy = nsp_accuracy.result().numpy()

    batch_mlm_mask = tf.cast(batch_mlm_mask, dtype=tf.int32)
    index = tf.where(batch_mlm_mask == 1)
    x_predict = tf.math.argmax(mlm_predict, axis=-1)
    x_predict = tf.gather_nd(x_predict, index)
    x_real = tf.gather_nd(origin_x, index)
    mlm_accuracy = tf.keras.metrics.Accuracy()
    mlm_accuracy.update_state(x_predict, x_real)
    mlm_accuracy = mlm_accuracy.result().numpy()

    return nsp_accuracy, mlm_accuracy
