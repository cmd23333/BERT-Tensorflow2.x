import os
from collections import Counter
import tensorflow as tf
import numpy as np
from config import Config


class Tokenizer:
    def __init__(self, config):
        with open(config['Vocabulary_File_Path'], 'r', encoding='utf-8') as f:
            self.dict = ['CLS', 'SEP', 'MASK', 'PAD', 'UNK'] + eval(f.read())
        self.word2id = {self.dict[i]: i for i in range(len(self.dict))}
        self.id2word = {i: self.dict[i] for i in range(len(self.dict))}

        self._token_start_id = self.word2id['CLS']
        self._token_end_id = self.word2id['SEP']
        self._token_mask_id = self.word2id['MASK']

    def encode(self, text):
        token_ids = [self._token_start_id] + [self.word2id[char] for char in text] + [self._token_end_id]
        segment_ids = [0 for char in text]
        return token_ids, segment_ids

    def decode(self, ids):
        return self.id2word[ids]


class Corpus:

    def __init__(self, config):
        self.config = config
        self.vocab2id, self.id2vocab = self.generate_vocabulary()
        self.data = []

    def generate_vocabulary(self):

        if os.path.exists(self.config['Vocabulary_File_Path']):
            with open(self.config['Vocabulary_File_Path'], 'r', encoding='utf-8') as f:
                vocabs = eval(f.read())
        else:
            with open(self.config['Corpus_File_Path'], 'r', encoding='utf-8') as f:
                corpus_ = f.read()
            vocabs_with_frequency = Counter(corpus_).most_common()
            vocabs = [word for (word, freq) in vocabs_with_frequency if
                      freq > self.config['Character_Frequency_Threshold']]
            with open(self.config['Vocabulary_File_Path'], 'w', encoding='utf-8') as f:
                f.write(str(vocabs))

        # 如果 config['Character_Frequency_Threshold'] <= 0, 那么训练时就不会用到'UNK'这一token.
        vocabs = ['CLS', 'SEP', 'MASK', 'PAD', 'UNK'] + vocabs
        vocab2id = dict(zip(vocabs, list(range(len(vocabs)))))
        id2vocab = dict(zip(list(range(len(vocabs))), vocabs))

        print('Vocabulary Size = {}'.format(len(vocab2id)))

        return vocab2id, id2vocab

    def make_and_parse_passages(self):
        with open(self.config['Corpus_File_Path'], 'r', encoding='utf-8') as f:
            corpus_ = f.readlines()
        line_index = 0
        while line_index < len(corpus_):
            # print('{}/{}'.format(line_index, len(corpus_)))
            if corpus_[line_index].startswith('名字1') or corpus_[line_index].startswith('地区:'):
                start_index = line_index
                while line_index + 1 < len(corpus_) and \
                        not corpus_[line_index + 1].startswith('名字1') and \
                        not corpus_[line_index + 1].startswith('地区:'):
                    line_index += 1
                line_index += 1
                passage = ''.join(corpus_[start_index: line_index])
                passage = passage.strip('')  # 没记错的话每段故事后面都有两个空格
                if passage.startswith('名字'):
                    name_index = passage.index('名字2')
                    region_index = passage.index('所属地区')
                    story_index = passage.index('故事')
                    nickname = passage[4:name_index].strip('\n')  # 无双剑姬
                    name = passage[name_index + 4:region_index].strip('\n')  # 菲奥娜
                    region = passage[region_index + 5:story_index].strip('\n')  # 德玛西亚
                    story = passage[story_index + 3:].strip('\n')  # ...
                    for i in range(10):
                        yield story

                elif passage.startswith('地区'):
                    '''
                    地区:比尔吉沃特
                    地区简介:比尔吉沃特是走私贩、劫掠者和不义之徒的避难港湾。在这里，富可敌国或是家破人亡都只在转瞬之间。对于那些逃避审判、债务和迫害的人，这个城市能让他们重获新生，因为在比尔吉沃特的蜿蜒街路上，没人会在乎你的过去。话虽如此，每当拂晓之际，粗心大意之人都会漂在港湾中，钱袋空空，喉头见血......
                    虽然比尔吉沃特是极其危险的地方，但这里也充满了机遇，不受到任何政府、法令、和道德的制约束缚。无论是来路不正的海克斯科技，还是当地黑帮的俯首听命，只要你出得起钱，一概唾手可得。
                    '''
                    story_index = passage.index('地区简介')
                    name = passage[3:story_index].strip('\n')
                    story = passage[story_index + 5:].strip('\n')
                    for i in range(10):
                        yield story

    def make_bert_data(self):
        passages = self.make_and_parse_passages()
        for story in passages:
            sentences = story.strip('。').split('。')
            if len(sentences) == 1:
                # 目前没有这种情况。
                print('这段话只有一句话，搞不了。')
                continue
            for i in range(len(sentences) - 1):
                one_sample = [self.vocab2id['CLS']]
                for char in sentences[i].strip('\n'):
                    if char in self.vocab2id:
                        one_sample.append(self.vocab2id[char])
                    else:
                        one_sample.append(self.vocab2id['UNK'])
                one_sample.append(self.vocab2id['。'])  # 按句号切割的，把句号还回来
                one_sample.append(self.vocab2id['SEP'])
                for char in sentences[i + 1].strip('\n'):
                    if char in self.vocab2id:
                        one_sample.append(self.vocab2id[char])
                    else:
                        one_sample.append(self.vocab2id['UNK'])
                one_sample.append(self.vocab2id['。'])  # 按句号切割的，把句号还回来
                one_sample.append(self.vocab2id['SEP'])

                separate_index = one_sample.index(self.vocab2id['SEP'])
                neg_one_sample = [self.vocab2id['CLS']] + one_sample[separate_index+1:] + one_sample[1:separate_index+1]
                if len(one_sample) < self.config['Max_Sequence_Length']:
                    one_sample += [self.vocab2id['PAD']] * (self.config['Max_Sequence_Length'] - len(one_sample))
                    neg_one_sample += [self.vocab2id['PAD']] * (self.config['Max_Sequence_Length'] - len(neg_one_sample))
                self.data.append(one_sample[:self.config['Max_Sequence_Length']])
                self.data.append(neg_one_sample[:self.config['Max_Sequence_Length']])

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


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.corpus = Corpus(config)
        print('start load corpus.')
        self.corpus.make_bert_data()
        print('load corpus done.')
        self.data = self.corpus.data
        self.batch_size = self.config['Batch_Size']
        assert self.batch_size % 2 == 0, '数据都是一个next sentence, 一个previous sentence, 所以batch size要是偶数'
        self.mask_token_id = self.corpus.vocab2id['MASK']

    def __len__(self):
        # 很显然，最后不满batch size个的sample被丢弃了
        return len(self.data) // self.batch_size

    def make_mask_language_model_data(self, batch_token_id):
        """
        对token ids做随机15%的mask。[MASK]的id默认为2。
        batch_token_id: [batch, max_seq_len]
        """
        batch_size = len(batch_token_id)
        # [PAD]的token id是3
        batch_padding_mask = (np.array(batch_token_id) != self.corpus.vocab2id['PAD']).astype(int)
        # 计算一个批次中每个句子除了[PAD]的长度
        batch_real_seq_lens = np.sum(batch_padding_mask, axis=1)
        # 计算一个批次中的每个句子，分别由多少单词要被[MASK]替换掉
        batch_mask_word_num = np.ceil(batch_real_seq_lens * self.config['Mask_Rate']).astype(int)

        mask_position = []
        for i in range(batch_size):
            # 不遮挡第一个CLS和最后一个SEP
            position = np.random.choice(a=np.arange(1, batch_real_seq_lens[i]-1), size=batch_mask_word_num[i],
                                        replace=False)
            mask_position.append(np.sum(np.eye(self.config['Max_Sequence_Length'])[position], axis=0))

        mask_position = np.array(mask_position)
        # 把该mask的地方都变成mask的token id
        mask_value_matrix = mask_position * self.mask_token_id
        inputs_mask = (mask_position == 0).astype(int)
        # 其他输入不变，被MASK的位置被替换为 mask token id
        batch_token_id_after_mlm = (batch_token_id * inputs_mask + mask_value_matrix).astype(int)
        return batch_token_id_after_mlm, mask_position

    def make_segment_inputs(self, batch_token_id):
        segment_inputs = []
        for i in range(len(batch_token_id)):
            try:
                separate_index = batch_token_id[i].index(self.corpus.vocab2id['SEP'])
            except ValueError:
                separate_index = len(batch_token_id[i]) - 1
                # 这句话太长了，256个都属于这句话。目测是！？等没用来做分割。
            one_segment_inputs = [0] * (separate_index + 1) + [1] * (self.config['Max_Sequence_Length'] - separate_index - 1)
            segment_inputs.append(one_segment_inputs)
        segment_inputs = np.array(segment_inputs)
        return segment_inputs

    def make_padding_mask(self, batch_token_id):
        batch_padding_mask = (np.array(batch_token_id) == self.corpus.vocab2id['PAD']).astype(int)
        return batch_padding_mask

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x, batch_mlm_mask = self.make_mask_language_model_data(batch_data)
        segment_x = self.make_segment_inputs(batch_data)
        padding_mask = self.make_padding_mask(batch_data)
        is_next_sentence = np.array([1, 0] * (self.batch_size // 2))
        shuffle = np.random.choice(np.arange(self.batch_size), size=self.batch_size, replace=False)
        # 返回6个东西，分别是mask后的句子,分句位置,补零位置; 原始句子,mask的位置,是否是下一句
        batch_x, batch_segment, batch_padding_mask = batch_x[shuffle], segment_x[shuffle], padding_mask[shuffle]
        origin_x, batch_mlm_mask, batch_y = np.array(batch_data)[shuffle], batch_mlm_mask[shuffle], is_next_sentence[shuffle]
        return batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y


if __name__ == '__main__':
    dataset = DataGenerator(Config)
    batch_x,  batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y = dataset[3]
    print(dataset.corpus.token_id_to_word_list(list(batch_x[0])))
    print(batch_y[0])
