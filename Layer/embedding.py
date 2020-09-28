import tensorflow as tf


class SegmentEmbedding(tf.keras.layers.Layer):
    """
    表示断句信息的编码
    就第一个句子全是0，第二个句子全是1,依此类推
    输入只有一个句子的话，加不加segment embedding都是一样的
    """
    def __init__(self,
                 **kwargs):
        super(SegmentEmbedding, self).__init__(**kwargs)

    def call(self, inputs):
        """有需要的话可以自定义编写一下"""
        return inputs


class PositionEmbedding(tf.keras.layers.Layer):
    """
    位置编码
    bert的位置编码是直接加可学习的embedding矩阵[max_seq_len, emb_len]，然后和输入相加。
    transformer的位置编码是根据公式计算的，计算完后和其他embedding相加
    除这两种外，还有相对位置编码，transformXL和XLNet里有用到
    """
    def __init__(self, embedding_size, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.position_embedding = self.add_weight(name='pos_embedding',
                                        shape=(input_shape[1], self.embedding_size),
                                        initializer=tf.keras.initializers.get('zeros'))

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embedding = self.position_embedding[:seq_len]
        pos_embedding = tf.expand_dims(pos_embedding, 0)
        pos_embedding = tf.tile(pos_embedding, [batch_size, 1, 1])
        return pos_embedding + inputs


class TokenEmbedding(tf.keras.layers.Layer):
    """
    token embedding
    """
    def __init__(self, vocab_size, embedding_size, **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        super(TokenEmbedding, self).build(input_shape)
        self.token_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                    output_dim=self.embedding_size)

    def call(self, inputs):
        output = self.token_embedding(inputs)
        return output


class TransTokenEmbedding(tf.keras.layers.Layer):
    """
    使用token embedding的转置矩阵使bert的输出变回到token id
    """
    def __init__(self, token_embedding_weights, **kwargs):
        super(TransTokenEmbedding, self).__init__(**kwargs)
        self.token_embedding_weights = token_embedding_weights

    def build(self, input_shape):
        super(TransTokenEmbedding, self).build(input_shape)
        self.transpose_token_emb = tf.transpose(self.token_embedding_weights)
        self.units = tf.shape(self.transpose_token_emb)[1]   # vocab数
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros')

    def call(self, inputs):
        outputs = tf.keras.backend.dot(inputs, self.transpose_token_emb)
        outputs = tf.keras.backend.bias_add(outputs, self.bias)
        outputs = tf.nn.softmax(outputs)
        return outputs
