from input_generator import InputGenerator
import numpy as np
from dnn import FeedForwardNetwork, compute_ce_loss
from ctc import compute_forced_alignment, compute_softmax
from utils import sequence_mask
import pickle

print("Starting training")

#configs
BATCH_SIZE = 5 
CONTEXT_LENGTH = 7
SUBSAMPLING_RATE = 3
JSON_FILE = "train_data.json"
FEATURE_DIM = 83
SPLICED_DIM = FEATURE_DIM * (2 * CONTEXT_LENGTH + 1)  
OUTPUT_DIM = 29
HIDDEN_LAYER_WIDTH = 500
NUM_HIDDEN_LAYERS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9 
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 500

def train_asr():
    generator = InputGenerator(
        json_file=JSON_FILE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        context_length=CONTEXT_LENGTH,
        subsampling_rate=SUBSAMPLING_RATE
    )
    
    total_utterances = len(generator.utterances)
    steps_per_epoch = total_utterances // BATCH_SIZE
    print(f"\nDataset statistics:")
    print(f"Total utterances: {total_utterances}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps for {NUM_EPOCHS} epochs: {steps_per_epoch * NUM_EPOCHS}")
    
    print("Checking for existing checkpoints...")
    try:
        with open('asr_model_latest.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
            model = checkpoint['model']
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
            velocities = checkpoint.get('velocities', None)  
            print(f"Resumed from epoch {start_epoch}, step {step}")
    except FileNotFoundError:
        print("No checkpoint found. Initializing new model...")
        model = FeedForwardNetwork(
            din=SPLICED_DIM,
            dout=OUTPUT_DIM,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            hidden_layer_width=HIDDEN_LAYER_WIDTH
        )
        start_epoch = 0
        step = 0
        velocities = {
            'weights': [np.zeros_like(w) for w in model.weights],
            'biases': [np.zeros_like(b) for b in model.biases]
        }
        print("New model initialized")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        current_epoch = generator.epoch
        
        while current_epoch == generator.epoch:
            batch = generator.next()
            
            _, features_list, labels_list = zip(*batch)
            
            max_input_len = max(len(feat) for feat in features_list)
            max_output_len = max(len(lab) for lab in labels_list)
            
            features_padded = np.zeros((BATCH_SIZE, max_input_len, features_list[0].shape[1]))
            labels_padded = np.zeros((BATCH_SIZE, max_output_len), dtype=np.int64)
            
            input_lengths = []
            output_lengths = []
            for i, (feat, lab) in enumerate(zip(features_list, labels_list)):
                features_padded[i, :len(feat)] = feat
                labels_padded[i, :len(lab)] = lab
                input_lengths.append(len(feat))
                output_lengths.append(len(lab))
            
            reshaped_features = features_padded.reshape(BATCH_SIZE * max_input_len, -1).T

            logits, hidden = model.forward(reshaped_features)
            
            logits_reshaped = logits.T.reshape(BATCH_SIZE, -1, OUTPUT_DIM)
            
            best_costs, alignments = compute_forced_alignment(
                -logits_reshaped,
                np.array(input_lengths),
                labels_padded,
                np.array(output_lengths)
            )
            
            loss_mask = sequence_mask(np.array(input_lengths), max(input_lengths)).astype(np.float32)
            loss_mask_flat = loss_mask.flatten()
            
            loss, grad_out = compute_ce_loss(
                logits,
                alignments.flatten(),
                loss_mask_flat
            )
            
            w_grads, b_grads = model.backward(
                reshaped_features,
                hidden,
                grad_out,
                loss_mask_flat
            )
            
            for i in range(len(model.weights)):
                velocities['weights'][i] = MOMENTUM * velocities['weights'][i] - LEARNING_RATE * w_grads[i]
                velocities['biases'][i] = MOMENTUM * velocities['biases'][i] - LEARNING_RATE * b_grads[i]

                model.weights[i] += velocities['weights'][i]
                model.biases[i] += velocities['biases'][i]
            
            if step % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'model': model,
                    'epoch': epoch,
                    'step': step,
                    'velocities': velocities 
                }
                with open('asr_model_latest.pkl', 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"Step {step}, Loss: {loss:.4f}")
            
            step += 1
        
        checkpoint = {
            'model': model,
            'epoch': epoch + 1,
            'step': step,
            'velocities': velocities
        }
        with open('asr_model_latest.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Completed epoch {epoch + 1}, Loss: {loss:.4f}")

    model.save_model('asr_model.pkl')
    print("Training completed!")

if __name__ == "__main__":
    print("Training started")
    train_asr()