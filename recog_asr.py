import numpy as np
import dnn
import ctc
from input_generator import InputGenerator, CharacterTokenizer
import editdistance
import pickle
import json


din = 83 * 15 
dout = 29
num_hidden_layers = 2
hidden_layer_width = 600

print("Loading model...")
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
network.restore_model('asr_model.pkl')
print("Model loaded successfully")

print("Initializing dev set generator...")
with open('dev_data.json', 'r') as f:
    dev_data = json.load(f)
utterance_ids = list(dev_data['utts'].keys())
total_utterances = len(utterance_ids)

dev_generator = InputGenerator(
    'dev_data.json', 
    batch_size=1,
    shuffle=False, 
    context_length=7,
    subsampling_rate=3
)

tokenizer = CharacterTokenizer()
progress_interval = total_utterances // 10
total_chars = 0
total_edit_distance = 0

print(f"\nStarting recognition of {total_utterances} utterances...")
for i, utterance in enumerate(utterance_ids):
    batch = dev_generator.next()
    uttid, features, ground_truth = batch[0]
    
    t, d = features.shape
    features = np.reshape(features, [t, d]).T
    _, probs = network.predict(features)
    log_probs = np.log(probs)
    
    hypothesis = ctc.beam_search(log_probs.T[None, :, :], np.array([t]))[0]
    
    ground_truth_str = tokenizer.IdsToString(ground_truth)
    hypothesis_str = tokenizer.IdsToString(hypothesis)
    
    edit_dist = editdistance.eval(ground_truth_str, hypothesis_str)
    total_chars += len(ground_truth_str)
    total_edit_distance += edit_dist
    
    if (i + 1) % progress_interval == 0:
        print(f"Progress: {((i + 1) / total_utterances * 100):.0f}% ({i + 1}/{total_utterances} utterances)")

char_error_rate = total_edit_distance / total_chars
print(f"\nEvaluation Complete:")
print(f"Character Error Rate: {(char_error_rate * 100):.4f}%")