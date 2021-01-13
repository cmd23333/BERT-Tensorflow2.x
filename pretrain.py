import os
import tensorflow as tf
from BertLayer import Bert
from Data.data import DataGenerator
from Loss.loss import BERT_Loss
from Loss.utils import calculate_pretrain_task_accuracy
from config import Config
from datetime import datetime


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = Bert(Config)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
loss_fn = BERT_Loss()
dataset = DataGenerator(Config)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(Config['Saved_Weight']))
manager = tf.train.CheckpointManager(checkpoint, directory=Config['Saved_Weight'], max_to_keep=5)
log_dir = os.path.join(Config['Log_Dir'], datetime.now().strftime("%Y-%m-%d"))
writer = tf.summary.create_file_writer(log_dir)

EPOCH = 10000
for epoch in range(EPOCH):
    for step in range(len(dataset)):
        batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y = dataset[step]
        with tf.GradientTape() as t:
            nsp_predict, mlm_predict, sequence_output = model((batch_x, batch_padding_mask, batch_segment),
                                                              training=True)
            nsp_loss, mlm_loss = loss_fn((mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y))
            nsp_loss = tf.reduce_mean(nsp_loss)
            mlm_loss = tf.reduce_mean(mlm_loss)
            loss = nsp_loss + mlm_loss
        gradients = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        nsp_acc, mlm_acc = calculate_pretrain_task_accuracy(nsp_predict, mlm_predict, batch_mlm_mask, origin_x, batch_y)

        if step % 100 == 0:
            print(
                'Epoch {}, step {}, loss {:.4f}, mlm_loss {:.4f}, mlm_acc {:.4f}, nsp loss {:.4f}, nsp_acc {:.4f}'.format(
                    epoch, step, loss.numpy(),
                    mlm_loss.numpy(),
                    mlm_acc,
                    nsp_loss.numpy(), nsp_acc
                    ))

        with writer.as_default():
            tf.summary.scalar('train_loss', loss.numpy(), step=epoch * len(dataset) + step)
            tf.summary.scalar('mlm_loss', mlm_loss.numpy(), step=epoch * len(dataset) + step)
            tf.summary.scalar('nsp_loss', nsp_loss.numpy(), step=epoch * len(dataset) + step)
            tf.summary.scalar('mlm_accuracy', mlm_acc, step=epoch * len(dataset) + step)
            tf.summary.scalar('nsp_accuracy', nsp_acc, step=epoch * len(dataset) + step)

    path = manager.save(checkpoint_number=epoch)
    print('model saved to %s' % path)
