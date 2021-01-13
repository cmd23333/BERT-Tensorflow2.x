import tensorflow as tf
from Layer.transformer import Transformer
from Layer.embedding import EmbeddingProcessor


class Bert(tf.keras.Model):
    """
    可以用过修改transformer的层数，还有输入的最大长度，来调节模型的大小
    可以设置segment embedding，对有断句的输入做embedding
    可以设置attention mask，在特定任务中可能会对attention做特定的mask
    """

    def __init__(self, config, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.vocab_size = config['Vocab_Size']
        self.embedding_size = config['Embedding_Size']
        self.max_seq_len = config['Max_Sequence_Length']
        self.segment_size = config['Segment_Size']
        self.num_transformer_layers = config['Num_Transformer_Layers']
        self.num_attention_heads = config['Num_Attention_Heads']
        self.intermediate_size = config['Intermediate_Size']
        self.initializer_range = config['Initializer_Variance']
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.embedding = EmbeddingProcessor(vocab_szie=self.vocab_size, embedding_size=self.embedding_size,
                                            max_seq_len=self.max_seq_len,
                                            segment_size=self.segment_size, )
        self.transformer_blocks = [Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)] * self.num_transformer_layers
        self.nsp_predictor = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None):
        batch_x, batch_mask, batch_segment = inputs

        x = self.embedding((batch_x, batch_segment))
        for i in range(self.num_transformer_layers):
            x = self.transformer_blocks[i](x, mask=batch_mask, training=training)

        first_token_tensor = x[:, 0, :]  # [batch_size ,hidden_size]
        nsp_predict = self.nsp_predictor(first_token_tensor)
        mlm_predict = tf.matmul(x, self.embedding.token_embedding.embeddings, transpose_b=True)
        sequence_output = x

        return nsp_predict, mlm_predict, sequence_output
