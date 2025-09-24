import json
from collections import Counter
import lmstudio as lms
from lmstudio import LlmLoadModelConfig
import time
import os

#get access token from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM model
config = LlmLoadModelConfig(
    context_length=16384,
)

analyse_model_name = os.getenv("ANALYSE_MODEL_NAME", "qwen3-4b-instruct-2507")
model = lms.llm(analyse_model_name, config=config)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

# Load the transcription data
@measure_time
def load_transcription_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Determine the interviewee's speaker based on total speaking time or word count
@measure_time
def determine_interviewee_speaker(data):
    speaker_stats = analyze_speakers(data)
    # Find the speaker with the most total words (or total time)
    interviewee_speaker = max(speaker_stats, key=lambda speaker: speaker_stats[speaker]["total_words"])
    return interviewee_speaker

# Analyze speaker statistics
@measure_time
def analyze_speakers(data):
    speaker_stats = {}
    for segment in data:
        speaker = segment["speaker"]
        duration = segment["end"] - segment["start"]
        words = len(segment["text"].split())

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"segments": 0, "total_time": 0, "total_words": 0}

        speaker_stats[speaker]["segments"] += 1
        speaker_stats[speaker]["total_time"] += duration
        speaker_stats[speaker]["total_words"] += words

    return speaker_stats

# Analyze text data
@measure_time
def analyze_text(data):
    all_text = " ".join([segment["text"] for segment in data])
    word_counts = Counter(all_text.split())
    return word_counts

# Analyze segment timing
@measure_time
def analyze_segments(data):
    durations = [segment["end"] - segment["start"] for segment in data]
    avg_duration = sum(durations) / len(durations) if durations else 0
    longest_segment = max(durations) if durations else 0
    shortest_segment = min(durations) if durations else 0
    return avg_duration, longest_segment, shortest_segment

# Extract names and roles using LLM
@measure_time
def extract_names_and_roles(data, interviewee_speaker):
    #find the interviewer name from first segment that is not the interviewee
    interviewer_speaker = next(segment["speaker"] for segment in data if segment["speaker"] != interviewee_speaker)
    first_interviewee_segments = [segment for segment in data if segment["speaker"] == interviewee_speaker][:5]
    first_interviewer_segments = [segment for segment in data if segment["speaker"] == interviewer_speaker][:5]
    

    prompt = f"""
You will be given transcript segments from the beginning of an interview between two speakers. 
Your task is to identify the full names and roles (interviewer or interviewee) of each speaker based on what they say.

Instructions:
- Use indicative phrases such as "My name is..." or "I am..." to extract names.
- Look for capitalized tokens that resemble real human names (i.e. capitalized first and last names).
- Do not guess names if they are not explicitly mentioned — leave as null or "[UNKNOWN]".
- Respond only in valid JSON format as shown below, and omit any extra commentary or explanation.

Input:

Segments from the interviewee:
{ " ".join([segment["text"] for segment in first_interviewee_segments]) }

Segments from the interviewer:
{ " ".join([segment["text"] for segment in first_interviewer_segments]) }

Output:
Respond only with a JSON object in the following format:
{{
  "interviewee": {{
    "name": "[Full name]",
    "role": "Interviewee"
  }},
  "interviewer": {{
    "name": "[Full name]",
    "role": "Interviewer"
  }}
}}
"""
    
    print("Sending prompt to LLM for names and roles...")
    response = model.respond(prompt)
    print(f"Model used: {response.model_info.display_name}")
    print(f"Response time: {response.stats.stop_reason} seconds")
    print(f"Tokens used: {response.stats.predicted_tokens_count}")
    print(f"Time to first token: {response.stats.time_to_first_token_sec} seconds")
    print("LLM response received.")
    try:
        names_and_roles = json.loads(response.content)
    except json.JSONDecodeError:
        print("Failed to parse JSON response from LLM.")
        names_and_roles = {
            "interviewee": {"name": "Unknown", "role": "Interviewee"},
            "interviewer": {"name": "Unknown", "role": "Interviewer"}
        }
    return names_and_roles

# Extract themes using LLM
@measure_time
def extract_themes(data, interviewee_speaker):
    interviewee_responses = " ".join(
        [segment["text"] for segment in data if segment["speaker"] == interviewee_speaker]
    )

    prompt = f"""
You are a qualitative research assistant helping to conduct thematic analysis.

Analyze the following interview transcript and extract key themes:

1. Identify prominent themes.
2. For each theme, provide:
   - A short title
   - A brief explanation
   - 1–2 supporting quotes

Transcript:
{interviewee_responses}

Structure your output like this:

Theme 1:
- Title: [Theme title]
- Description: [Short explanation]
- Quotes: "[Quote 1]" | "[Quote 2]"

Be concise but insightful. Focus on issues, current practices, organizational structures, and challenges.

Only provide the structured output as shown above. Do not include any additional reasoning, explanations, or commentary.
"""

    print("Sending prompt to LLM...")
    response = model.respond(prompt)
    
    print(f"Model used: {response.model_info.display_name}")
    print(f"Response time: {response.stats.stop_reason} seconds")
    print(f"Tokens used: {response.stats.predicted_tokens_count}")
    print(f"Time to first token: {response.stats.time_to_first_token_sec} seconds")
    print("LLM response received.")
    return response.content

# Main function
@measure_time
def main():
    annotations_folder = os.getenv("COMBINED_ANNOTATIONS_FOLDER") # Output folder for finished annotations

    # Walk through all subfolders in the annotations folder
    for root, dirs, files in os.walk(annotations_folder):
        for file in files:
            if file == "transcription_segments.json":  # Only process JSON files with this name
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Load the transcription data
                data = load_transcription_data(file_path)

                # Get the folder path of the JSON file
                folder_path = os.path.dirname(file_path)
                output_file = os.path.join(folder_path, "analysis_results.txt")

                # Open the output file for writing
                with open(output_file, "w", encoding="utf-8") as f_out:
                    # Speaker analysis
                    speaker_stats = analyze_speakers(data)
                    f_out.write("Speaker Statistics:\n")
                    for speaker, stats in speaker_stats.items():
                        f_out.write(f"  {speaker}:\n")
                        f_out.write(f"    Segments: {stats['segments']}\n")
                        f_out.write(f"    Total Time: {stats['total_time']:.2f} seconds\n")
                        f_out.write(f"    Total Words: {stats['total_words']}\n")
                    f_out.write("\n")

                    # Determine the interviewee's speaker
                    interviewee_speaker = determine_interviewee_speaker(data)
                    f_out.write(f"Determined Interviewee Speaker: {interviewee_speaker}\n\n")

                    # Text analysis
                    word_counts = analyze_text(data)
                    f_out.write("Most Common Words:\n")
                    for word, count in word_counts.most_common(10):
                        f_out.write(f"  {word}: {count}\n")
                    f_out.write("\n")

                    # Segment timing analysis
                    avg_duration, longest_segment, shortest_segment = analyze_segments(data)
                    f_out.write("Segment Timing:\n")
                    f_out.write(f"  Average Duration: {avg_duration:.2f} seconds\n")
                    f_out.write(f"  Longest Segment: {longest_segment:.2f} seconds\n")
                    f_out.write(f"  Shortest Segment: {shortest_segment:.2f} seconds\n\n")
                    
                    # Extract names and roles using LLM
                    names_and_roles = extract_names_and_roles(data, interviewee_speaker)
                    f_out.write("Identified Participants:\n")
                    f_out.write(f"  Interviewee: {names_and_roles['interviewee']['name']} ({names_and_roles['interviewee']['role']})\n")
                    f_out.write(f"  Interviewer: {names_and_roles['interviewer']['name']} ({names_and_roles['interviewer']['role']})\n\n")

                    # Extract themes using LLM
                    themes = extract_themes(data, interviewee_speaker)
                    f_out.write("Extracted Themes:\n")
                    f_out.write(themes)
                    f_out.write("\n")

                print(f"Analysis results saved to: {output_file}")

if __name__ == "__main__":
    main()