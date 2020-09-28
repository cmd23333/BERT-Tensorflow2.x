import tensorflow as tf
from BertLayer import BertLayer
from Layer.embedding import TransTokenEmbedding
from Data.data import DataGenerator
from Loss.mln_loss import MLMLoss
from config import Config


class BERT(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.bert = BertLayer(config)
        self.predict = tf.keras.layers.Dense(config['Vocab_Size'])

    def call(self, inputs):
        outputs = self.bert(inputs)
        logits = tf.nn.softmax(self.predict(outputs))
        return logits


loss_fn = MLMLoss()
model = BERT(Config)
optimizer = tf.keras.optimizers.Adam()
dataset = DataGenerator(Config)

EPOCH = 10000

for i in range(EPOCH):
    for step in range(len(dataset)):
        batch_x, label, batch_mask = dataset[step]
        with tf.GradientTape() as t:
            logits = model(batch_x)
            loss = loss_fn([label, logits, batch_mask])
        gradients = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('step:', i, 'loss:', loss.numpy())

