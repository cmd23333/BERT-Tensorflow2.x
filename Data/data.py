import os
import tensorflow as tf
import numpy as np
from config import Config


class Dataset:

    def __init__(self, config):
        super(Dataset, self).__init__()
        self.config = config
        self.max_seqlen = config['Max_Sequence_Len']

        if os.path.exists(config['Vocabulary_File_Path']):
            with open(config['Vocabulary_File_Path'], 'r', encoding='gbk') as f:
                vocab = eval(f.read())
        else:
            vocab = self.generate_vocabulary()

        self.vocab2id = dict(zip(vocab, list(range(len(vocab)))))
        self.id2vocab = dict(zip(list(range(len(vocab))), vocab))

    def generate_vocabulary(self):
        with open(self.config['Corpus_File_Path'], 'r') as f:
            corpus = f.read()
        vocab = list(set(corpus))
        with open(self.config['Vocabulary_File_Path'], 'w') as f:
            f.write(str(vocab))
        return vocab


    def truncate(self, string):
        """
        截断太长的输入, 因为开头要加上[CLS], 中间和结尾要加上[SEP], 所以实际允许的序列最大长度是 Max_Sequence_Len - 3
        """
        if len(string) > self.max_seqlen - 3:
            return string[:self.max_seqlen - 3]
        else:
            return string

    @staticmethod
    def decorate(word_list):
        """
        在输入的序列头部加上[CLS]
        """
        return ['[CLS]'] + word_list

    def padding(self, word_list):
        """
        给长度不够的句子补上 [PAD]
        """
        return word_list + ['[PAD]'] * (self.max_seqlen - len(word_list))

    def word_list_to_token_id(self, word_list):
        """
        把单词(包括文字和特殊token)序列转化成 token id 序列
        """
        token_id_list = []
        for word in word_list:
            if word in self.vocab2id:
                token_id_list.append(self.vocab2id[word])
            else:
                token_id_list.append(self.vocab2id['UNK'])
        return token_id_list

    def token_id_to_word_list(self, token_id_list):
        """
        把token id序列转换回原始的单词列表。
        """
        word_list = []
        for token_id in token_id_list:
            if token_id in self.id2vocab:
                word_list.append(self.id2vocab[token_id])
            else:
                word_list.append('[UNK]')
        return word_list

    def string_to_token_id(self, string):
        """
        整合一下上面几个函数
        """
        string = list(self.truncate(string))
        word_list = self.decorate(string)
        word_list = self.padding(word_list)
        token_id = self.word_list_to_token_id(word_list)
        return token_id

    def token_id_to_string(self, token_id):
        """
        整合一下上面几个函数
        """
        word_list = self.word_list_to_token_id(token_id)
        string = ''
        for word in word_list:
            if word == '[CLS]':
                continue
            if word == '[PAD]':
                break
            string += word
        return string


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.max_seq_len = config['Max_Sequence_Len']
        self.batch_size = config['Batch_Size']
        self.mask_rate = config['Mask_Rate']

        self.data = ['精准而优雅', '我渴望有价值的对手', '刺穿他们', '优雅的对手']
        self.preprocess = PreProcessing(config)
        self.mask_token_id = self.preprocess.vocab2id['[MASK]']

    def __len__(self):
        return len(self.data)//self.batch_size

    def add_mlm_mask(self, batch_token_id):
        """
        对token ids做随机15%的mask。[MASK]的id默认为4。
        """
        batch_size = len(batch_token_id)
        # [PAD]的token id是0
        batch_padding_mask = (np.array(batch_token_id) > 0).astype(int)
        # 计算一个批次中每个句子除了[PAD]的长度
        batch_real_seq_lens = np.sum(batch_padding_mask, axis=1)
        # 计算一个批次中的每个句子，分别由多少单词要被[MASK]替换掉
        batch_mask_word_num = np.ceil(batch_real_seq_lens * self.mask_rate).astype(int)

        mask_position = []
        for i in range(batch_size):
            if batch_real_seq_lens[i] == 2:  # 除了[CLS]以外，只有一个字符，不做[MASK]了
                position = np.random.choice(a=np.arange(1, batch_real_seq_lens[i]), size=0, replace=False)
                mask_position.append(np.sum(np.eye(self.max_seq_len)[position], axis=0))
            else:
                position = np.random.choice(a=np.arange(1, batch_real_seq_lens[i]), size=batch_mask_word_num[i], replace=False)
                mask_position.append(np.sum(np.eye(self.max_seq_len)[position], axis=0))

        mask_position = np.array(mask_position)
        # 把该mask的地方都变成mask的token id
        mask_value_matrix = mask_position * self.mask_token_id
        inputs_mask = (mask_position == 0).astype(int)
        # 其他输入不变，被MASK的位置被替换为 mask token id
        batch_token_id_after_mlm = (batch_token_id * inputs_mask + mask_value_matrix).astype(int)
        return batch_token_id_after_mlm, mask_position

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_token_id = [self.preprocess.string_to_token_id(string) for string in batch_data]
        batch_x, batch_mask = self.add_mlm_mask(batch_token_id)
        return [batch_x, batch_token_id, batch_mask]

