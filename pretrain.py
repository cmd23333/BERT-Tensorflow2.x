import tensorflow as tf
from BertLayer import BertLayer
from Data.data import DataGenerator
from Loss.mln_loss import BertLoss
from config import Config
from datetime import datetime
import os


class BERT_pretrain(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(BERT_pretrain, self).__init__(**kwargs)
        self.bert = BertLayer(config)
        self.mlm_predictor = tf.keras.layers.Dense(config['Vocab_Size'])
        self.nsp_predictor = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None):
        outputs = self.bert(inputs, training=training)
        mlm_pred = tf.nn.softmax(self.mlm_predictor(outputs))
        nsp_pred = tf.nn.softmax(self.nsp_predictor(outputs[:, 0, :]))
        return mlm_pred, nsp_pred


loss_fn = BertLoss()
model = BERT_pretrain(Config)
optimizer = tf.keras.optimizers.Adam()
dataset = DataGenerator(Config)
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=Config['Saved_Weight'], max_to_keep=5)
log_dir = os.path.join(Config['Log_Dir'], datetime.now().strftime("%Y-%m-%d"))
writer = tf.summary.create_file_writer(log_dir)

EPOCH = 100

for epoch in range(EPOCH):
    for step in range(len(dataset)):
        batch_x, batch_segment, batch_padding_mask, origin_x, batch_mlm_mask, batch_y = dataset[step]
        with tf.GradientTape() as t:
            mlm_pred, nsp_pred = model([batch_x, batch_padding_mask, batch_segment], training=True)
            mlm_loss, nsp_loss, mlm_acc, nsp_acc = loss_fn([mlm_pred, nsp_pred, origin_x, batch_mlm_mask, batch_y])
            loss = mlm_loss + nsp_loss
        gradients = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % 100 == 0:
            print('Epoch {}, step {}, loss {:.4f}'.format(epoch, step, loss.numpy()))
            print('mlm_loss {:.4f}, mlm_acc {:.2f}'.format(mlm_loss.numpy(), mlm_acc.numpy()))
            print('nsp_loss {:.4f}, nsp_acc {:.2f}'.format(nsp_loss.numpy(), nsp_acc.numpy()))

        with writer.as_default():
            tf.summary.scalar('train_loss', loss.numpy(), step=epoch*len(dataset) + step)
            tf.summary.scalar('mlm_loss', mlm_loss.numpy(), step=epoch*len(dataset) + step)
            tf.summary.scalar('nsp_loss', nsp_loss.numpy(), step=epoch*len(dataset) + step)
            tf.summary.scalar('mlm_accuracy', mlm_acc.numpy(), step=epoch*len(dataset) + step)
            tf.summary.scalar('nsp_accuracy', nsp_acc.numpy(), step=epoch*len(dataset) + step)

    path = manager.save(checkpoint_number=epoch)
    print('model saved to %s' % path)
