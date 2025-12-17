import json
import os
from collections import Counter
from itertools import chain

class Vocab:
    def __init__(self, path, src_language, tgt_language, min_freq=5):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.src_s2i = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.src_i2s = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}
        self.tgt_s2i = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx, self.bos_token: self.bos_idx, self.eos_token: self.eos_idx}
        self.tgt_i2s = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token, self.bos_idx: self.bos_token, self.eos_idx: self.eos_token}

        self.build_vocab(path)

    def load_data(self, path):
        files = ["small-train.json", "small-dev.json", "small-test.json"]
        data = []
        for file in files:
            full_path = os.path.join(path, file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    data.extend(json.load(f))
        return data

    def build_vocab(self, path):
        data = self.load_data(path)
        
        src_tokens = [item[self.src_language].split() for item in data]
        tgt_tokens = [item[self.tgt_language].split() for item in data]
        
        src_counter = Counter(chain.from_iterable(src_tokens))
        tgt_counter = Counter(chain.from_iterable(tgt_tokens))

        # Xây dựng từ điển Source (tiếng Anh)
        for token, count in src_counter.items():
            if count >= self.min_freq and token not in self.src_s2i:
                idx = len(self.src_s2i)
                self.src_s2i[token] = idx
                self.src_i2s[idx] = token
        
        # Xây dựng từ điển Target (tiếng Việt)
        for token, count in tgt_counter.items():
            if count >= self.min_freq and token not in self.tgt_s2i:
                idx = len(self.tgt_s2i)
                self.tgt_s2i[token] = idx
                self.tgt_i2s[idx] = token
        
        self.src_vocab_size = len(self.src_s2i)
        self.tgt_vocab_size = len(self.tgt_s2i)

    def encode(self, text, is_target=False):
        tokens = text.split()
        s2i = self.tgt_s2i if is_target else self.src_s2i
        unk_idx = self.unk_idx

        indices = [s2i.get(token, unk_idx) for token in tokens]
        
        if is_target:
            indices = [self.bos_idx] + indices + [self.eos_idx]
            
        return indices

    def decode(self, indices, is_target=False):
        i2s = self.tgt_i2s if is_target else self.src_i2s
        tokens = [i2s.get(idx, self.unk_token) for idx in indices]
        return " ".join(tokens)
