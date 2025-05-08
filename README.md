# Speech Serve

*A CTC-based Speech Recognition System*

## 🎙️ Overview

**Speech Serve** is a speech recognition system built using **Connectionist Temporal Classification (CTC)** and a Deep Neural Network (DNN) acoustic model. It supports training on variable-length speech data, performing forced alignment, and evaluating transcription quality using beam search decoding. The system computes **Character Error Rate (CER)** as its main evaluation metric.

---

## 🚀 Features

- 📦 CTC-based training loop with pseudo-labels via forced alignment  
- 🔁 Support for variable-length inputs with padding and masking  
- 🧠 DNN-based acoustic model with log-probability output  
- 🔍 Beam search decoding for transcription  
- 📏 Edit distance evaluation (CER) using `editdistance` package  
- 🕐 Frame-level forced alignment with timestamp mapping

---

## 🧪 Training Pipeline

1. Load a mini-batch of utterances (with context padding and subsampling).
2. Forward pass through the DNN to obtain per-frame token posteriors.
3. Compute forced alignment using log-probabilities.
4. Treat aligned labels as pseudo-ground-truth and calculate masked loss.
5. Backpropagate and update model weights.
6. Save model checkpoints periodically for recovery.

---

## 📊 Evaluation Pipeline

- Load and preprocess utterances with the same settings as training.
- Run the DNN to get log-probabilities per frame.
- Use beam search (`ctc.py`) to decode the most likely token sequence.
- Compute **Character Error Rate (CER)** using the `editdistance` package.

---

## 🧾 Forced Alignment

This feature maps each token in a ground-truth transcript to specific frames in the audio.  
Example:  
> Token `[14, 15, 18]` corresponds to "NOR" between frames 22–28  
> At 30ms/frame → "NOR" is spoken between 0.63s and 0.84s

---

## 📁 File Structure

- `train_asr.py` — Training script  
- `recog_asr.py` — Evaluation script (prints final CER)  
- `test_forced_alignment.py` — Forced alignment test script  
- `asr_model.pkl` — Saved model checkpoint  
- Dependencies from previous projects:  
  - Input generator (Project 1)  
  - DNN forward logic (Project 3)  
  - Forced alignment & search (Project 2)

---

## 💻 Requirements

- Python 3.x  
- TensorFlow (CPU version)  
- `editdistance` package:  
  ```bash
  pip install editdistance
  ```

---

## 🏁 Output Example

Transcript:  
```
NOR IS MISTER QUILTER’S MANNER LESS INTERESTING THAN HIS MATTER
```

Predicted timing for "NOR":  
```
Frame Range: 22–28 → Time: 0.63s to 0.84s
```

---

## 📌 Notes

- Model performance may be limited compared to state-of-the-art systems.
- This implementation focuses on the CTC training and forced alignment mechanism.
- There’s room to improve accuracy using better tokenization, model architecture, and regularization.


