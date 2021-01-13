import tensorflow as tf


class EmbeddingProcessor(tf.keras.layers.Layer):
    def __init__(self, vocab_szie, embedding_size=768, max_seq_len=512,
                 segment_size=2, hidden_dropout_prob=0.0, initializer_range=0.02,
                 **kwargs):
        super(EmbeddingProcessor, self).__init__(**kwargs)
        self.vocab_size = vocab_szie
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.segment_size = segment_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.token_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                         output_dim=self.embedding_size,
                                                         embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                             self.initializer_range),
                                                         name="token_embedding",
                                                         dtype=tf.float32
                                                         )
        self.segment_embedding = tf.keras.layers.Embedding(input_dim=self.segment_size,
                                                           output_dim=self.embedding_size,
                                                           embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                               self.initializer_range),
                                                           name="segment_embedding",
                                                           dtype=tf.float32)
        self.positional_embedding = self.add_weight(name='positional_embeddings',
                                                    shape=(self.max_seq_len, self.embedding_size),
                                                    initializer=tf.keras.initializers.TruncatedNormal(
                                                        self.initializer_range),
                                                    dtype=tf.float32)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        self.output_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob, dtype=tf.float32)
        super(EmbeddingProcessor, self).build(input_shape)

    def call(self, inputs):
        input_ids, segment_ids = inputs
        seq_length = input_ids.shape[1]
        token_token_embeddings = self.token_embedding(input_ids)  # [batch_size, seq_len, d]
        token_segment_embeddings = self.segment_embedding(segment_ids)  # [batch_size, seq_len, d]
        token_positional_embeddings = tf.expand_dims(self.positional_embedding[:seq_length, :], axis=0)  # [1,seq_len,d]

        output = token_token_embeddings + token_segment_embeddings + token_positional_embeddings
        output = self.output_layer_norm(output)
        output = self.output_dropout(output)
        return output
