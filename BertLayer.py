import tensorflow as tf
from Layer.transformer import Transformer
from Layer.embedding import PositionEmbedding, TokenEmbedding


class BertLayer(tf.keras.layers.Layer):
    """
    可以用过修改transformer的层数，还有输入的最大长度，来调节模型的大小
    可以设置segment embedding，对有断句的输入做embedding
    可以设置attention mask，在特定任务中可能会对attention做特定的mask
    """
    def __init__(self, config, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.vocab_size = config['Vocab_Size']
        self.embedding_size = config['Embedding_Size']
        self.num_transformer_layers = config['Num_Transformer_Layers']
        self.num_attention_heads = config['Num_Attention_Heads']
        self.intermediate_size = config['Intermediate_Size']
        self.initializer_range = config['Initializer_Range']
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)
        self.token_embedding = TokenEmbedding(vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        self.position_embedding = PositionEmbedding(embedding_size=self.embedding_size)
        self.ln = tf.keras.layers.LayerNormalization()
        self.transformer_block = Transformer(heads=self.num_attention_heads, head_size=int(self.embedding_size/self.num_attention_heads), intermediate_size=self.intermediate_size)

    def call(self, inputs):
        # 构建padding mask
        padding_mask = tf.cast(tf.keras.backend.greater(inputs, 0), 'float32')
        # token embedding
        token_embedding = self.token_embedding(inputs)
        # position embedding
        position_embedding = self.position_embedding(token_embedding)
        # layer norm
        x = self.ln(position_embedding)
        # transformer blocks
        # 这里是经过同一个transformer多次，参照albert
        for i in range(self.num_transformer_layers):
            x = self.transformer_block(x, att_mask=padding_mask)
        return x
