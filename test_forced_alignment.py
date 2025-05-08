import numpy as np
import dnn
import ctc
import input_generator


# Define model.
din = 83 * 15
# id 0 is reserved for blank.
dout = 29
num_hidden_layers = 2
hidden_layer_width= 600
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
network.restore_model('asr_model.pkl')

dev_iter = input_generator.InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=7, subsampling_rate=3)
dev_iter.next()
uttid, x, y = dev_iter.next()[0]
print(f'uttid={uttid}')
gt_str = dev_iter.tokenizer.IdsToString(y)
print(f'transcript={gt_str}')

t, d = x.shape
# dnn takes input with the format [din, time].
x = np.transpose(np.reshape(x, [t, d]))
pred_labels, probs = network.predict(x)
probs = np.transpose(probs)[None, :, :]

_, alignment = ctc.compute_forced_alignment(- np.log(probs), np.array([t]), np.array(y)[None, :], np.array([len(y)]))
print(alignment)

"""
alignment=
[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 14 14 15
   0 18 18 18 28  9  0  0  0  0  0 19 19 28 28 13  9 19 19 20  5 18  0 28
  17 21  9  0 12 20  0  5 18 18 27  0 19 19  0 28 28 13  1  0 14  0 14  5
  18 18 18 18  0 28 28  0 12  5 19  0  0 19 28 28 28  9 14 14  0 20 20  5
  18 18 18  5 19 19 19 20 20  9 14  7  0  0 28 20  8  1  0 14 14  0  0 28
   8  9  0  0  0  0 19 19  0  0  0  0 28 13  1  1 20  0 20 20  5 18 18 18
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]


NOR          tokens 14,15,18                           frames 22-28     time 0.63-0.84
IS           tokens 9,19                               frames 30-37     time 0.87-1.11
MISTER       tokens 13,9,19,20,5,18                    frames 40-46     time 1.17-1.38
QUILTER'S    tokens 17,21,9,12,20,5,18,27,19           frames 49-62     time 1.44-1.86
MANNER       tokens 13,1,14,14,5,18                    frames 66-76     time 1.95-2.28
LESS         tokens 12,5,19,19                         frames 81-86     time 2.40-2.58
INTERESTING  tokens 9,14,20,5,18,5,19,20,9,14,7        frames 90-108    time 2.67-3.24
THAN         tokens 20,8,1,14                          frames 112-117   time 3.33-3.51
HIS          tokens 8,9,19                             frames 121-128   time 3.60-3.84
MANNER       tokens 13,1,20,20,5,18                    frames 134-144   time 3.99-4.32

"""