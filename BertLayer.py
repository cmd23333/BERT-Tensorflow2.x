import tensorflow as tf
from Layer.transformer import Transformer
from Layer.embedding import PositionEmbedding, SegmentEmbedding, TokenEmbedding


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
        self.initializer_range = config['Initializer_Variance']
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)
        self.token_embedding = TokenEmbedding(vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        self.position_embedding = PositionEmbedding(embedding_size=self.embedding_size)
        self.segment_embedding = SegmentEmbedding(embedding_size=self.embedding_size)
        self.transformer_block = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads, dff=self.intermediate_size)

    def call(self, inputs, training=None):
        batch_x, batch_mask, batch_segment = inputs

        token_embedding = self.token_embedding(batch_x)
        position_embedding = self.position_embedding(batch_x)
        segment_embedding = self.segment_embedding(batch_segment)

        x = token_embedding + position_embedding + segment_embedding
        # transformer blocks
        # 这里是经过同一个transformer多次，参照albert
        for i in range(self.num_transformer_layers):
            x = self.transformer_block(x, mask=batch_mask, training=training)
        return x
