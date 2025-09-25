# Interview Analysis Toolkit

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated pipeline for transcribing, analyzing, and extracting insights from interview recordings using AI-powered speech recognition and natural language processing.

## What this project does

This toolkit transforms raw interview recordings into structured, research-ready analysis by:

- **Transcribing** audio/video files with speaker identification using WhisperX
- **Analyzing** conversations to identify themes, speaker roles, and key insights using large language models
- **Processing** files individually or in batches for efficient workflow
- **Generating** structured output suitable for qualitative research analysis

Perfect for researchers, journalists, and professionals who need to process and analyze interview data at scale.

## Why this project is useful

**Save time and effort:**

- Automates hours of manual transcription work
- Processes multiple interviews simultaneously
- Provides consistent analysis across all recordings

**Enhance analysis quality:**

- Identifies speakers automatically with diarization
- Extracts themes and insights using advanced AI models
- Generates statistics on speaking patterns and engagement

**Research-ready output:**

- Structured JSON and text formats
- Speaker role identification (interviewer vs interviewee)
- Timestamped segments for detailed analysis

## How to get started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for performance)
- [LM Studio](https://lmstudio.ai/) or compatible LLM interface for analysis

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd dippa
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with your settings:

   ```bash
   # Required: Set your WhisperX access token
   ACCESS_TOKEN="your_huggingface_token"

   # Optional: Customize model names and paths
   WHISPER_MODEL_NAME="Finnish-NLP/whisper-large-finnish-v3-ct2"
   ANALYSE_MODEL_NAME="qwen3-4b-instruct-2507"
   ```

### Quick start

**Process multiple interviews:**

```bash
# 1. Place your MP4 files in the mp4_files/ folder
# 2. Transcribe all files
python transcribe_multiple.py

# 3. Analyze all transcriptions
python analyse_multiple.py
```

**Process a single interview:**

```bash
# Set file paths in .env first
python transcribe_singular.py
python analyse_singular.py
```

### Example output

The analysis generates structured files containing:

```
Speaker Statistics:
- SPEAKER_00: 1,247 words (15.2 minutes)
- SPEAKER_01: 3,891 words (28.7 minutes)

Identified Roles:
- Interviewee: Maria Virtanen (IT Administrator)
- Interviewer: Research Assistant

Key Themes:
1. User Feedback Challenges
   "We struggle to get meaningful feedback from end users..."

2. System Implementation Barriers
   "The biggest issue is resistance to change..."
```

## Project structure

```
dippa/
├── transcribe_multiple.py    # Batch audio transcription
├── transcribe_singular.py    # Single file transcription
├── analyse_multiple.py       # Batch analysis of transcripts
├── analyse_singular.py       # Single file analysis
├── .env.example             # Configuration template
├── mp4_files/              # Input audio/video files
├── chunks/                 # Temporary audio segments
├── annotations/            # Individual analysis results
└── annotations_combined/   # Merged analysis outputs
```

## Configuration options

### Models

**Transcription models:**

- `Finnish-NLP/whisper-large-finnish-v3-ct2` (Finnish)
- `openai/whisper-large-v3` (Multi-language)

**Analysis models:**

- `qwen3-4b-instruct-2507` (Recommended)
- `llama-3.1-8b-instruct`
- Any LM Studio compatible model

### Environment variables

| Variable             | Description                    | Default                                    |
| -------------------- | ------------------------------ | ------------------------------------------ |
| `ACCESS_TOKEN`       | HuggingFace token for WhisperX | Required                                   |
| `WHISPER_MODEL_NAME` | Speech-to-text model           | `Finnish-NLP/whisper-large-finnish-v3-ct2` |
| `ANALYSE_MODEL_NAME` | LLM for analysis               | `qwen3-4b-instruct-2507`                   |
| `MP4_FOLDER`         | Input files directory          | `./mp4_files`                              |

## Where to get help

**Documentation:**

- Check function docstrings in source files for detailed API information
- Review `.env.example` for all configuration options

**Common issues:**

- **Memory errors**: Reduce batch size or use smaller models
- **Poor transcription**: Ensure clear audio quality and correct language model
- **Analysis failures**: Verify LM Studio is running and model is loaded

**Support:**

- Open an issue for bugs or feature requests
- Check existing issues for solutions to common problems

## Who maintains this project

This project is maintained as part of Master's thesis research in Social and Health Informatics.

**Maintainer:** [Your Name] - [contact@email.com]

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For major changes, open an issue first to discuss proposed modifications.

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Ready to start?** Place your interview files in `mp4_files/` and run `python transcribe_multiple.py`!
