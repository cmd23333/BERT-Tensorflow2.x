import os
import numpy as np
import tensorflow as tf
from BertLayer import Bert
from config import Config
from Data.data import Tokenizer


def dist(x, y):
    """距离函数（默认用欧氏距离）
    可以尝试换用内积或者cos距离，结果差不多。
    """
    return np.sqrt(((x - y)**2).sum())


# 文本编码
def make_segment(tokenizer, text, thres):
    token_ids, segment_ids = tokenizer.encode(text)
    length = len(token_ids) - 2


    batch_token_ids = np.array([token_ids] * (2 * length - 1))
    batch_segment_ids = np.zeros_like(batch_token_ids)
    batch_mask_ids = np.zeros_like(batch_token_ids)

    for i in range(length):
        if i > 0:
            batch_token_ids[2 * i - 1, i] = tokenizer._token_mask_id
            batch_token_ids[2 * i - 1, i + 1] = tokenizer._token_mask_id
        batch_token_ids[2 * i, i + 1] = tokenizer._token_mask_id

    _, _, vectors = model.predict([batch_token_ids, batch_mask_ids, batch_segment_ids])
    word_token_ids = [[token_ids[1]]]
    for i in range(1, length):
        d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
        d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
        d = (d1 + d2) / 2
        if d >= thres:
            word_token_ids[-1].append(token_ids[i + 1])
        else:
            word_token_ids.append([token_ids[i + 1]])

    words = [tokenizer.decode(ids) for ids in word_token_ids]
    return words


model = Bert(Config)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(Config['Saved_Weight']))

