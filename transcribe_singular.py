import whisperx
import gc
import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Configuration
access_token = os.getenv("ACCESS_TOKEN")
whisper_model_name = os.getenv("WHISPER_MODEL_NAME")
audio_file = os.getenv("AUDIO_FILE")
annotations_folder = os.getenv("ANNOTATIONS_FOLDER")

os.makedirs(annotations_folder, exist_ok=True)

device = "cuda"
batch_size = 4
compute_type = "float16"

# Function to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

# Function to load the WhisperX model
@measure_time
def load_model():
    return whisperx.load_model(whisper_model_name, device, compute_type=compute_type)

# Function to transcribe audio
@measure_time
def transcribe_audio(model, audio_file):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size), audio
        
    return result

# Function to align transcription
@measure_time
def align_transcription(result, audio):
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    aligned_txt_file = os.path.join(annotations_folder, "aligned_transcription.txt")
    with open(aligned_txt_file, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f'Start: {segment["start"]:.2f}, End: {segment["end"]:.2f}, Speaker: {segment.get("speaker", "N/A")}, Text: {segment["text"]}\n')

    return result
# Function to perform diarization
@measure_time
def diarize_audio(audio):
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=access_token, device=device)
    result = diarize_model(audio, max_speakers=2)
    
    diarization_txt_file = os.path.join(annotations_folder, "diarization.txt")
    
    with open(diarization_txt_file, "w", encoding="utf-8") as f:
        f.write("Diarization Result:\n")
        f.write("Segment\tLabel\tSpeaker\tStart\tEnd\n")
        for idx, row in enumerate(result.iterrows()):
            segment = row[1]
            f.write(
                f"{idx}\t[{segment['start']:.3f} --> {segment['end']:.3f}]\t"
                f"{segment['label']}\t{segment['speaker']}\t"
                f"{segment['start']:.3f}\t{segment['end']:.3f}\n"
            )

    return result

# Function to assign speaker labels
@measure_time
def assign_speakers(diarize_segments, result):
    return whisperx.assign_word_speakers(diarize_segments, result)

# Function to save transcription to JSON
@measure_time
def save_transcription_to_json(result, file_path):
    segment_level_data = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "N/A"),
            "text": seg["text"]
        }
        for seg in result["segments"]
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(segment_level_data, f, indent=4, ensure_ascii=False)
    print(f"Segment-level transcription saved to: {file_path}")

# Main script
if __name__ == "__main__":
    # clear annotations folder
    for file in os.listdir(annotations_folder):
        file_path = os.path.join(annotations_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    # Load model
    model = load_model()

    # Transcribe audio
    result, audio = transcribe_audio(model, audio_file)

    # Align transcription
    result = align_transcription(result, audio)

    # Perform diarization
    diarize_segments = diarize_audio(audio)

    # Assign speaker labels
    result = assign_speakers(diarize_segments, result)

    # Save transcription to JSON
    transcription_json_file = os.path.join(annotations_folder, "transcription_segments.json")
    save_transcription_to_json(result, transcription_json_file)

    # Clear GPU memory
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()