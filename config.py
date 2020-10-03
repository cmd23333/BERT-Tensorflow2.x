import os

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

Config = {
    'Corpus_File_Path': os.path.join(PROJECT_PATH, 'Data/lol_corpus.txt'),
    'Vocabulary_File_Path': os.path.join(PROJECT_PATH, 'Data/vocab.txt'),  # 词表存放位置
    'Log_Dir': os.path.join(PROJECT_PATH, 'Logs'),
    'Saved_Weight': os.path.join(PROJECT_PATH, 'Saved_Weight'),
    'Character_Frequency_Threshold': 3,
    'Batch_Size': 32,
    'Max_Sequence_Length': 128,  # 最大长度
    'Mask_Rate': 0.15,
    'Vocab_Size': 2367,
    'Embedding_Size': 256,
    'Num_Transformer_Layers': 6,
    'Num_Attention_Heads': 8,
    'Intermediate_Size': 1024,
    'Initializer_Variance': 0.02,  # 权重初始化方差，默认0.02
}
