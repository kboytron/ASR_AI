from flask import Flask, request, jsonify
import numpy as np
import os
import librosa
from input_generator import InputGenerator, CharacterTokenizer
from ctc import compute_forced_alignment, beam_search
from dnn import FeedForwardNetwork

app = Flask(__name__, static_folder="frontend", static_url_path="")

# Load model and tokenizer
dnn = FeedForwardNetwork.load("asr_model.pkl")  # Your model file
tokenizer = CharacterTokenizer()

@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_blob = request.files["audio"].read()  # Raw audio bytes from browser
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_blob)

    # Process audio
    features = process_audio(audio_path)
    log_probs = dnn.forward(features)
    transcription_ids = beam_search(log_probs)
    transcription = tokenizer.ids_to_string(transcription_ids)
    alignment = compute_forced_alignment(log_probs, tokenizer.string_to_ids(transcription))

    os.remove(audio_path)
    return jsonify({
        "transcription": transcription,
        "alignment": alignment.tolist()
    })

def process_audio(audio_path):
    # Convert .wav to features matching your DNN input
    y, sr = librosa.load(audio_path, sr=16000)  # 16kHz
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=83)  # 83 dims from Project 1
    log_mel = np.log(mel + 1e-10).T  # Frames x 83
    ig = InputGenerator(None, 1, False, 7, 3)  # No JSON needed for single audio
    return ig.splice_and_subsample(log_mel, 7, 3)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)