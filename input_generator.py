import json
import random
import kaldiio
import numpy as np

class CharacterTokenizer:
    def __init__(self):
        self.char_to_id = {chr(65+i): i+1 for i in range(26)}
        self.char_to_id[' '] = 28
        self.char_to_id["'"] = 27
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def StringToIds(self, text):
        return [self.char_to_id[c] for c in text.upper() if c in self.char_to_id]
    
    def IdsToString(self, ids):
        return ''.join(self.id_to_char[id] for id in ids if id in self.id_to_char)

def splice_and_subsample(X, C, r):
    T, d = X.shape
    spliced = []
    
    for i in range(T):
        left_context = [X[max(0, i - j)] for j in range(C, 0, -1)]
        current_frame = [X[i]]
        right_context = [X[min(T - 1, i + j)] for j in range(1, C + 1)]
        spliced_frame = np.concatenate(left_context + current_frame + right_context)
        spliced.append(spliced_frame)
    
    spliced = np.array(spliced)
    subsampled = spliced[::r, :]
    
    return subsampled

class InputGenerator:
    def __init__(self, json_file, batch_size, shuffle, context_length, subsampling_rate):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
        self.subsampling_rate = subsampling_rate
        self.epoch = 0
        self.step_within_epoch = 0
        self.total_num_steps = 0
        self.tokenizer = CharacterTokenizer()
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.utterances = self.data['utts']
        self.N = len(self.utterances)
        self.indices = list(range(self.N))
        if self.shuffle:
            random.shuffle(self.indices)

    def next(self):
        if self.step_within_epoch == 0 and self.shuffle:
            random.shuffle(self.indices)

        start = self.step_within_epoch * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]

        if len(batch_indices) < self.batch_size:
            batch_indices += random.sample(self.indices, self.batch_size - len(batch_indices))

        batch = []
        for idx in batch_indices:
            utterance_key = list(self.utterances.keys())[idx]
            utterance = self.utterances[utterance_key]
            utt_id = utterance_key
            features = kaldiio.load_mat(utterance['feat'])
            features = splice_and_subsample(features, self.context_length, self.subsampling_rate)
            labels = self.tokenizer.StringToIds(utterance['text'])
            batch.append((utt_id, features, labels))

        self.step_within_epoch += 1
        self.total_num_steps += 1
        if self.step_within_epoch * self.batch_size >= self.N:
            self.epoch += 1
            self.step_within_epoch = 0

        return batch