import whisperx
import gc
import os
import json
import time
import pandas as pd
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Configuration

#get access token from .env file
from dotenv import load_dotenv
load_dotenv()

access_token = os.getenv("ACCESS_TOKEN")
if not access_token:
    raise ValueError("ACCESS_TOKEN not found in environment variables.")

device = "cuda"

mp4_folder = os.getenv("MP4_FOLDER") # Input folder containing MP4 files
chunks_folder = os.getenv("CHUNKS_FOLDER") # Temporary folder for audio chunks
annotations_folder = os.getenv("ANNOTATIONS_FOLDER") # Output folder for finished annotations
combined_annotations_folder = os.getenv("COMBINED_ANNOTATIONS_FOLDER") # Output folder for combined annotations

whisper_batch_size = 8 # Number of audio samples to process in a batch during transcription

compute_type = "float16"
chunk_len_in_secs = 6000.0  # Length of each chunk in seconds
whisper_model_name = os.getenv("WHISPER_MODEL_NAME", "Finnish-NLP/whisper-large-finnish-v3-ct2") # WhisperX model name

convert_to_wav = False # Convert MP4 files to WAV and split into chunks to avoid memory issues
perform_transcription_and_diarization = True # Perform transcription and diarization on chunks
combine_transcription_and_diarizations = True # Combine all transcriptions and diarizations into single files per interview

os.makedirs(chunks_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)
os.makedirs(combined_annotations_folder, exist_ok=True)

# Function to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    clear_folder(file_path)  # Recursively clear subdirectory
                    os.rmdir(file_path)  # Remove the now-empty subdirectory
            except Exception as e:
                print(f"Error clearing file {file_path}: {e}")

# Function to split audio into chunks
@measure_time
def split_audio_into_chunks(input_folder, output_folder, chunk_len_in_secs):
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            audio_path = os.path.join(input_folder, file)
            audio = AudioSegment.from_wav(audio_path)
            chunk_len_ms = chunk_len_in_secs * 1000  # Convert seconds to milliseconds
            base_name = os.path.splitext(file)[0]

            # Split audio into chunks
            for i, start_time in enumerate(range(0, len(audio), int(chunk_len_ms))):
                chunk = audio[start_time:start_time + chunk_len_ms]
                chunk_filename = f"{base_name}_chunk{i}.wav"
                chunk_path = os.path.join(output_folder, chunk_filename)
                chunk.export(chunk_path, format="wav")
                print(f"Created chunk: {chunk_path}")

            # Delete the original .wav file after splitting
            os.remove(audio_path)
            print(f"Deleted original file: {audio_path}")

# Function to convert MP4 to WAV and ensure correct format
@measure_time
def convert_mp4_to_wav(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_folder, file)
            wav_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".wav")
            
            # Extract audio from MP4
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(wav_path)
            print(f"Extracted audio from {file} to {wav_path}")
            
            # Convert to mono and 16 kHz
            audio = AudioSegment.from_wav(wav_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")
            print(f"Converted {file} to mono and 16 kHz: {wav_path}")

# Function to load the WhisperX model
@measure_time
def load_model():
    return whisperx.load_model(whisper_model_name, device, compute_type=compute_type)

# Function to transcribe audio
@measure_time
def transcribe_audio(model, audio_file):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=whisper_batch_size), audio
        
    return result

# Function to align transcription
@measure_time
def align_transcription(result, audio):
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    return result

# Function to perform diarization
@measure_time
def diarize_audio(audio):
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=access_token, device=device)
    result = diarize_model(audio, max_speakers=2)

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

# Function to combine transcriptions
@measure_time
def combine_transcriptions(annotations_folder, combined_annotations_folder):
    # Ensure the combined folder exists
    os.makedirs(combined_annotations_folder, exist_ok=True)
    print(f"Annotations folder: {annotations_folder}")
    print(f"Combined annotations folder: {combined_annotations_folder}")

    # Group subfolders by interview name
    interview_groups = {}
    for subfolder in os.listdir(annotations_folder):
        subfolder_path = os.path.join(annotations_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-folder: {subfolder_path}")
            continue  # Skip if it's not a folder

        # Extract the interview name (e.g., "interview1" from "interview1_chunk0")
        interview_name = subfolder.split("_chunk")[0]
        if interview_name not in interview_groups:
            interview_groups[interview_name] = []
        interview_groups[interview_name].append(subfolder_path)

    print(f"Grouped interview subfolders: {json.dumps(interview_groups, indent=4)}")

    # Process each interview group
    for interview_name, subfolders in interview_groups.items():
        print(f"Processing interview: {interview_name}")

        # Create a folder for the combined files of this interview
        interview_combined_folder = os.path.join(combined_annotations_folder, interview_name)
        os.makedirs(interview_combined_folder, exist_ok=True)

        # Initialize combined JSON data
        combined_json_data = []

        # Process each chunk in the interview
        for subfolder in sorted(subfolders):  # Sort subfolders to maintain chunk order
            print(f"Processing subfolder: {subfolder}")

            # Append unaligned transcription
            unaligned_file = os.path.join(subfolder, "unaligned_transcription.txt")
            combined_unaligned_file = os.path.join(interview_combined_folder, "unaligned_transcription.txt")
            if os.path.exists(unaligned_file):
                print(f"Found unaligned transcription: {unaligned_file}")
                with open(unaligned_file, "r", encoding="utf-8") as f_in, open(combined_unaligned_file, "a", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
                print(f"Appended unaligned transcription from {unaligned_file} to {combined_unaligned_file}")
            else:
                print(f"Warning: Unaligned transcription not found in {subfolder}")

            # Append aligned transcription
            aligned_file = os.path.join(subfolder, "aligned_transcription.txt")
            combined_aligned_file = os.path.join(interview_combined_folder, "aligned_transcription.txt")
            if os.path.exists(aligned_file):
                print(f"Found aligned transcription: {aligned_file}")
                with open(aligned_file, "r", encoding="utf-8") as f_in, open(combined_aligned_file, "a", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
                print(f"Appended aligned transcription from {aligned_file} to {combined_aligned_file}")
            else:
                print(f"Warning: Aligned transcription not found in {subfolder}")

            # Append diarization
            diarization_file = os.path.join(subfolder, "diarization.txt")
            combined_diarization_file = os.path.join(interview_combined_folder, "diarization.txt")
            if os.path.exists(diarization_file):
                print(f"Found diarization file: {diarization_file}")
                with open(diarization_file, "r", encoding="utf-8") as f_in, open(combined_diarization_file, "a", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
                print(f"Appended diarization from {diarization_file} to {combined_diarization_file}")
            else:
                print(f"Warning: Diarization file not found in {subfolder}")

            # Merge transcription segments (JSON)
            transcription_json_file = os.path.join(subfolder, "transcription_segments.json")
            combined_json_file = os.path.join(interview_combined_folder, "transcription_segments.json")
            if os.path.exists(transcription_json_file):
                print(f"Found transcription segments JSON: {transcription_json_file}")
                with open(transcription_json_file, "r", encoding="utf-8") as f:
                    chunk_json_data = json.load(f)
                    combined_json_data.extend(chunk_json_data)
                print(f"Merged transcription segments JSON from {transcription_json_file}")
            else:
                print(f"Warning: Transcription segments JSON not found in {subfolder}")

        # Save the combined JSON data
        if combined_json_data:  # Only save if there is data to save
            with open(combined_json_file, "w", encoding="utf-8") as f:
                json.dump(combined_json_data, f, indent=4, ensure_ascii=False)
            print(f"Saved combined transcription segments JSON to {combined_json_file}")
        else:
            print(f"No transcription segments JSON data to save for {interview_name}")
            
def clear_gpu_memory():
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
        
# Main script
if __name__ == "__main__":
    if convert_to_wav:
        clear_folder(chunks_folder)
        convert_mp4_to_wav(mp4_folder, chunks_folder)
        split_audio_into_chunks(chunks_folder, chunks_folder, chunk_len_in_secs)
            
    if perform_transcription_and_diarization:
        clear_folder(annotations_folder)
            
        # Load model
        whisper_model = load_model()
        
        def extract_chunk_number(filename):
            # Extract the numeric part of the chunk (e.g., "chunk10" -> 10)
            parts = filename.split("_chunk")
            if len(parts) > 1 and parts[1].split(".")[0].isdigit():
                return int(parts[1].split(".")[0])
            return 0  # Default to 0 if no chunk number is found
        
        sorted_files = sorted(
            [file for file in os.listdir(chunks_folder) if file.endswith(".wav")],
            key=extract_chunk_number,
        )
        
        for file in sorted_files:
            audio_path = os.path.join(chunks_folder, file)
            print(f"Processing file: {audio_path}")
            
            try:
                # Create a subfolder for the current interview
                interview_name = os.path.splitext(file)[0]  # Get the base name without extension
                interview_folder = os.path.join(annotations_folder, interview_name)
                os.makedirs(interview_folder, exist_ok=True)

                # Transcribe audio
                result, audio = transcribe_audio(whisper_model, audio_path)
                clear_gpu_memory()

                # Save unaligned transcription
                unaligned_txt_file = os.path.join(interview_folder, "unaligned_transcription.txt")
                with open(unaligned_txt_file, "w", encoding="utf-8") as f:
                    f.write("Unaligned Transcription:\n")
                    for seg in result["segments"]:
                        f.write(f"Start: {seg['start']:.2f}, End: {seg['end']:.2f}, Text: {seg['text']}\n")

                # Align transcription
                result = align_transcription(result, audio)
                clear_gpu_memory()

                # Save aligned transcription
                aligned_txt_file = os.path.join(interview_folder, "aligned_transcription.txt")
                with open(aligned_txt_file, "w", encoding="utf-8") as f:
                    f.write("Aligned Transcription:\n")
                    for seg in result["segments"]:
                        f.write(f"Start: {seg['start']:.2f}, End: {seg['end']:.2f}, Speaker: {seg.get('speaker', 'N/A')}, Text: {seg['text']}\n")

                # Perform diarization
                diarize_segments = diarize_audio(audio)
                clear_gpu_memory()

                # Save diarization result
                diarization_txt_file = os.path.join(interview_folder, "diarization.txt")
                with open(diarization_txt_file, "w", encoding="utf-8") as f:
                    f.write("Diarization Result:\n")
                    f.write("Segment\tLabel\tSpeaker\tStart\tEnd\n")
                    for idx, row in enumerate(diarize_segments.iterrows()):
                        segment = row[1]
                        f.write(
                            f"{idx}\t[{segment['start']:.3f} --> {segment['end']:.3f}]\t"
                            f"{segment['label']}\t{segment['speaker']}\t"
                            f"{segment['start']:.3f}\t{segment['end']:.3f}\n"
                        )

                # Assign speaker labels
                result = assign_speakers(diarize_segments, result)
                clear_gpu_memory()

                # Save transcription segments to JSON
                transcription_json_file = os.path.join(interview_folder, "transcription_segments.json")
                save_transcription_to_json(result, transcription_json_file)

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    
    if combine_transcription_and_diarizations:
        clear_folder(combined_annotations_folder)
        combine_transcriptions(annotations_folder, combined_annotations_folder)