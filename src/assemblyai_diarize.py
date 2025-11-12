import requests
import time
import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"


def upload_file_to_assemblyai(audio_path: str, quiet: bool = False) -> str:
    """
    Upload an audio file to AssemblyAI and get the upload URL.
    
    Args:
        audio_path (str): Path to the audio file
        quiet (bool): Whether to suppress output
        
    Returns:
        str: Upload URL for the audio file
    """
    if not ASSEMBLYAI_API_KEY:
        raise ValueError(
            "ASSEMBLYAI_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    
    if not quiet:
        print(f"Uploading {audio_path} to AssemblyAI...")
    
    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{ASSEMBLYAI_BASE_URL}/upload",
            headers=headers,
            data=f
        )
    
    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.status_code} - {response.text}")
    
    upload_url = response.json()["upload_url"]
    
    if not quiet:
        print("Upload complete!")
    
    return upload_url


def start_transcription(
    audio_url: str,
    speaker_labels: bool = True,
    language_code: Optional[str] = None,
    quiet: bool = False
) -> str:
    """
    Start a transcription job with AssemblyAI.
    
    Args:
        audio_url (str): URL of the audio file (from upload or direct URL)
        speaker_labels (bool): Whether to enable speaker diarization
        language_code (str, optional): Language code (e.g., 'en', 'es')
        quiet (bool): Whether to suppress output
        
    Returns:
        str: Transcript ID
    """
    if not ASSEMBLYAI_API_KEY:
        raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
    
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    
    data = {
        "audio_url": audio_url,
        "speaker_labels": speaker_labels
    }
    
    if language_code:
        data["language_code"] = language_code
    
    if not quiet:
        print("Starting transcription with speaker diarization...")
    
    response = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/transcript",
        json=data,
        headers=headers
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Transcription request failed: {response.status_code} - {response.text}")
    
    transcript_id = response.json()["id"]
    
    if not quiet:
        print(f"Transcription started (ID: {transcript_id})")
    
    return transcript_id


def poll_transcription(transcript_id: str, quiet: bool = False) -> Dict:
    """
    Poll the transcription status until it's complete.
    
    Args:
        transcript_id (str): The transcript ID
        quiet (bool): Whether to suppress output
        
    Returns:
        Dict: The complete transcription result
    """
    if not ASSEMBLYAI_API_KEY:
        raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
    
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    polling_endpoint = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"
    
    if not quiet:
        print("Waiting for transcription to complete...")
    
    while True:
        response = requests.get(polling_endpoint, headers=headers)
        
        if response.status_code != 200:
            raise RuntimeError(f"Polling failed: {response.status_code} - {response.text}")
        
        result = response.json()
        status = result["status"]
        
        if status == "completed":
            if not quiet:
                print("Transcription completed!")
            return result
        elif status == "error":
            error_msg = result.get("error", "Unknown error")
            raise RuntimeError(f"Transcription failed: {error_msg}")
        else:
            if not quiet:
                print(f"Status: {status}...")
            time.sleep(3)


def format_transcript_with_speakers(
    result: Dict,
    speaker_names: Optional[List[str]] = None
) -> str:
    """
    Format the transcription result with speaker labels.
    
    Args:
        result (Dict): AssemblyAI transcription result
        speaker_names (List[str], optional): List of speaker names to use instead of Speaker A, B, etc.
                                            Order should match the speaker labels.
        
    Returns:
        str: Formatted transcript with speaker labels and timestamps
    """
    utterances = result.get("utterances", [])
    
    if not utterances:
        # Fallback to full text if no utterances
        return result.get("text", "")
    
    # Build speaker mapping if names provided
    speaker_map = {}
    if speaker_names:
        # Get unique speakers from utterances
        unique_speakers = sorted(set(u["speaker"] for u in utterances))
        
        for i, speaker_label in enumerate(unique_speakers):
            if i < len(speaker_names):
                speaker_map[speaker_label] = speaker_names[i]
            else:
                speaker_map[speaker_label] = f"Speaker {speaker_label}"
    
    formatted_lines = []
    
    for utterance in utterances:
        speaker = utterance["speaker"]
        text = utterance["text"]
        start_time = utterance["start"] / 1000  # Convert ms to seconds
        end_time = utterance["end"] / 1000
        
        # Format timestamp as [MM:SS - MM:SS]
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)
        timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        
        # Get speaker label
        if speaker_map:
            speaker_label = speaker_map.get(speaker, f"Speaker {speaker}")
        else:
            speaker_label = f"Speaker {speaker}"
        
        formatted_lines.append(f"{timestamp} {speaker_label}: {text}")
    
    return "\n".join(formatted_lines)


def transcribe_with_assemblyai(
    audio_path: str,
    speaker_names: Optional[List[str]] = None,
    language_code: Optional[str] = None,
    quiet: bool = False
) -> str:
    """
    Complete transcription workflow with AssemblyAI speaker diarization.
    
    Args:
        audio_path (str): Path to the audio file
        speaker_names (List[str], optional): List of speaker names (e.g., ['Alice', 'Bob'])
        language_code (str, optional): Language code (e.g., 'en')
        quiet (bool): Whether to suppress output
        
    Returns:
        str: Formatted transcript with speakers and timestamps
    """
    # Validate file exists
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Upload file
    upload_url = upload_file_to_assemblyai(str(audio_path_obj), quiet=quiet)
    
    # Start transcription
    transcript_id = start_transcription(
        upload_url,
        speaker_labels=True,
        language_code=language_code,
        quiet=quiet
    )
    
    # Poll for completion
    result = poll_transcription(transcript_id, quiet=quiet)
    
    # Format output
    formatted_transcript = format_transcript_with_speakers(result, speaker_names)
    
    return formatted_transcript