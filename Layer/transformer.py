import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力"""
    def __init__(self, heads, head_size, key_size=None, initializer='glorot_uniform', **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.wq = tf.keras.layers.Dense(units=self.heads*self.key_size, kernel_initializer=self.kernel_initializer)
        self.wk = tf.keras.layers.Dense(units=self.heads*self.key_size, kernel_initializer=self.kernel_initializer)
        self.wv = tf.keras.layers.Dense(units=self.heads*self.head_size, kernel_initializer=self.kernel_initializer)
        self.wo = tf.keras.layers.Dense(units=self.out_dim, kernel_initializer=self.kernel_initializer)

    def call(self, q, k, v, att_mask=None):
        # 先按需计算mask，这里是可以个人定制的
        if att_mask is not None:
            pass

        # 输入的x值通过三个不同的dense之后得到q, k, v
        q = self.wq(q)  # [b, l, n*k]
        k = self.wk(k)  # [b, l, n*k]
        v = self.wv(v)  # [b, l, n*h]

        # b: batch_size, n: num of heads, l: seq_length,
        # k: key_size, h: head_size (不额外设定时，h = k)
        # reshape, [b, l, n*h] → [b, l, n, h]
        q = tf.reshape(q, [-1, tf.shape(q)[1], self.heads, self.key_size])
        k = tf.reshape(k, [-1, tf.shape(k)[1], self.heads, self.key_size])
        v = tf.reshape(v, [- 1, tf.shape(v)[1], self.heads, self.head_size])

        # attention：qk点积、padding_mask与att_mask、softmax
        a = tf.einsum('binh,bjnh->bnij', q, k) / self.key_size**0.5
        a = tf.nn.softmax(a)

        # softmax值与v做加权平均，输出形状设置成和输入类似
        o = tf.einsum('bnij,bjnh->binh', a, v)
        o = tf.reshape(o, (-1, tf.shape(o)[1], self.out_dim))
        o = self.wo(o)

        return o


class Transformer(tf.keras.layers.Layer):

    def __init__(self, heads, head_size, intermediate_size, initializer='glorot_uniform', **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.intermediate_size = intermediate_size  # feed-forward的隐层维度
        self.kernel_initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        super(Transformer, self).build(input_shape)
        self.mha = MultiHeadAttention(heads=self.heads, head_size=self.head_size)
        self.add = tf.keras.layers.Add()
        self.ln = tf.keras.layers.LayerNormalization()
        self.ff_dense1 = tf.keras.layers.Dense(units=self.intermediate_size,
                                               activation='relu',
                                               kernel_initializer=self.kernel_initializer)
        self.ff_dense2 = tf.keras.layers.Dense(units=self.heads * self.head_size,
                                               activation='relu',
                                               kernel_initializer=self.kernel_initializer)

    def call(self, inputs, att_mask=None):
        # 多头注意力
        x0 = self.mha(inputs, inputs, inputs, att_mask=att_mask)
        # 残差连接
        x0 = self.add([x0, inputs])
        # layer normalization
        x0 = self.ln(x0)
        # feed-forward 实际为两层dense，第二层dense输出维度可以设置，但一般和前面的输出一样，就取了n*h
        x1 = self.ff_dense1(x0)
        x1 = self.ff_dense2(x1)
        # 残差连接
        x1 = self.add([x1, x0])
        # layer normalization
        x1 = self.ln(x1)
        return x1


