import tensorflow as tf


class SegmentEmbedding(tf.keras.layers.Layer):
    """
    表示断句信息的编码
    就第一个句子全是0，第二个句子全是1,依此类推
    输入只有一个句子的话，加不加segment embedding都是一样的
    """
    def __init__(self, embedding_size, **kwargs):
        super(SegmentEmbedding, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def build(self, input_shape, **kwargs):
        # 如果有更多的句子连起来, 把下面的shape里的2改掉就好了
        super(SegmentEmbedding, self).build(input_shape)
        self.segment_embedding = tf.keras.layers.Embedding(input_dim=2,
                                                           output_dim=self.embedding_size)

    def call(self, inputs):
        output = self.segment_embedding(inputs)
        return output


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
        # 这边的input_shape[1]就是 max sequence length, 因为我的程序里每个批次的长度已经定死了,
        # 如果改我的代码时, 每个批次的seq length不固定, 应该让shape等于所有输入中seq length最大的数字
        super(PositionEmbedding, self).build(input_shape)
        self.position_embedding = self.add_weight(name='pos_embedding',
                                        shape=(input_shape[1], self.embedding_size),
                                        initializer=tf.keras.initializers.get('glorot_uniform'))

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embedding = self.position_embedding[:seq_len]
        pos_embedding = tf.expand_dims(pos_embedding, 0)
        pos_embedding = tf.tile(pos_embedding, [batch_size, 1, 1])
        return pos_embedding


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
