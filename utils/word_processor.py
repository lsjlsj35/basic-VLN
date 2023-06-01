# encoding: utf-8
import re
import string
from collections import defaultdict
import numpy as np


class Tokenizer:
    def __init__(self, vocab, seq_len=20, reserving_list=(',', '.')):
        self.vocab = vocab
        self.word2idx = {}
        self.seq_len = seq_len
        self.SPLIT_REG = re.compile(r'(\W+)')  # 按标点空格进行分割，保留标点空格
        self.rl = reserving_list
        for i, word in enumerate(vocab):
            self.word2idx[word] = i

    def split(self, sentence):
        tokens = []
        for word in [s.lower().strip() for s in self.SPLIT_REG.split(sentence.strip())]:
            if len(word) > 0 and not any(c in string.punctuation for c in word) or word in self.rl:
                tokens.append(word)
        return tokens

    def encode(self, sentence):
        enc = [self.word2idx['<START>']]
        word_list = self.split(sentence)
        for word in word_list:
            if word in self.word2idx:
                enc.append(self.word2idx[word])
            else:
                enc.append(self.word2idx['<UNK>'])
        enc.append(self.word2idx['<EOS>'])
        if len(enc) < self.seq_len:
            enc.extend([self.word2idx['<PAD>']]*(self.seq_len - len(enc)))
        return np.array(enc[:self.seq_len])

    def decode(self, enc):
        seq = []
        for idx in enc:
            if idx == self.word2idx['<PAD>']:
                break
            else:
                seq.append(self.vocab[idx])
        return " ".join(seq)


