import whisper
import argparse
from pathlib import Path
import concurrent.futures
from typing import List, Optional, Dict, Tuple
import sys
import os

# Add current directory to path for whisper-diarization imports
sys.path.append(str(Path(__file__).parent))

# Import AssemblyAI module
try:
    from assemblyai_diarize import transcribe_with_assemblyai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False

import subprocess

# Check if whisper-diarization files are available
WHISPER_DIARIZATION_AVAILABLE = (
    (Path(__file__).parent / "diarize.py").exists() and
    (Path(__file__).parent / "helpers.py").exists()
)

def transcribe_audio_with_timestamps(audio_path: str, model_name: str = "base", language: Optional[str] = None, quiet: bool = False) -> Dict:
    """
    Transcribe an audio file using OpenAI's Whisper model with timestamps and language detection.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use (tiny, base, small, medium, large)
        language (str, optional): Language code (e.g., 'en' for English). If None, auto-detects.
        
    Returns:
        Dict containing:
        - text: The full transcript
        - segments: List of segments with timestamps
        - language: Detected language
    """
    # Validate audio file exists
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load the model
    if not quiet:
        print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name)
    
    # Transcribe the audio with language detection
    if not quiet:
        print(f"Transcribing {audio_path}...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False  # Suppress progress output
    )
    
    return result

def transcribe_audio(audio_path: str, model_name: str = "base", quiet: bool = False) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        str: Transcribed text
    """
    # Validate audio file exists
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load the model
    if not quiet:
        print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name)
    
    # Transcribe the audio
    if not quiet:
        print(f"Transcribing {audio_path}...")
    result = model.transcribe(str(audio_path))
    
    return result["text"]

def format_transcription_with_timestamps(result: Dict) -> str:
    """
    Format transcription result with timestamps.
    
    Args:
        result (Dict): Whisper transcription result containing segments
        
    Returns:
        str: Formatted transcript with timestamps
    """
    formatted_lines = []
    
    for segment in result.get('segments', []):
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        # Format timestamp as [MM:SS - MM:SS]
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)
        
        timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        formatted_lines.append(f"{timestamp} {text}")
    
    return '\n'.join(formatted_lines)

def transcribe_with_speaker_diarization(audio_path: str, model_name: str = "base", quiet: bool = False) -> str:
    """
    Perform transcription with speaker diarization using whisper-diarization.
    
    Args:
        audio_path (str): Path to the audio/video file
        model_name (str): Whisper model to use
        quiet (bool): Whether to suppress output
        
    Returns:
        str: Formatted transcript with speakers and timestamps
    """
    if not WHISPER_DIARIZATION_AVAILABLE:
        error_msg = """
ERROR: whisper-diarization not available!

Required files (diarize.py, helpers.py) are missing.
Please check the installation.
"""
        if not quiet:
            print(error_msg)
        raise ImportError("whisper-diarization files not available")
    
    try:
        if not quiet:
            print("Performing transcription with speaker diarization...")
        
        # Get the directory containing the script files
        script_dir = Path(__file__).parent
        diarize_script = script_dir / "diarize.py"
        
        # Build command arguments with memory-conservative settings
        cmd = [
            "python", str(diarize_script),
            "-a", audio_path,
            "--whisper-model", "tiny",  # Use smaller model to reduce memory usage
            "--device", "cpu",
            "--batch-size", "1"  # Reduce batch size to minimize memory usage
        ]
        
        # Run the diarization script
        if quiet:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        else:
            result = subprocess.run(cmd, cwd=script_dir)
        
        if result.returncode != 0:
            error_output = result.stderr if hasattr(result, 'stderr') and result.stderr else "Unknown error"
            raise RuntimeError(f"Diarization script failed with return code {result.returncode}: {error_output}")
        
        # Read the generated transcript file
        audio_path_obj = Path(audio_path)
        transcript_file = audio_path_obj.with_suffix('.txt')
        
        if not transcript_file.exists():
            raise RuntimeError(f"Expected transcript file not found: {transcript_file}")
        
        transcript_content = transcript_file.read_text(encoding='utf-8-sig')
        
        if not quiet:
            print("Speaker diarization completed successfully!")
        
        return transcript_content
        
    except Exception as e:
        error_msg = f"""
ERROR in whisper-diarization: {e}

This could be due to:
1. Unsupported audio/video format
2. Missing system dependencies (ffmpeg)
3. Insufficient memory
4. Corrupted audio file
5. Script execution failure

Please check the file and try again.
"""
        if not quiet:
            print(error_msg)
        raise RuntimeError(f"Speaker diarization failed: {e}")

def process_file(file_path: Path, model_name: str) -> None:
    """Process a single audio/video file and save its transcription."""
    try:
        transcription = transcribe_audio(str(file_path), model_name)
        output_path = file_path.with_suffix('.txt')
        output_path.write_text(transcription, encoding="utf-8")
        print(f"Transcription saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_file_with_timestamps(file_path: Path, model_name: str) -> None:
    """Process a single audio/video file and save its transcription with timestamps."""
    try:
        result = transcribe_audio_with_timestamps(str(file_path), model_name)
        transcription = format_transcription_with_timestamps(result)
        output_path = file_path.with_suffix('.txt')
        output_path.write_text(transcription, encoding="utf-8")
        print(f"Transcription with timestamps saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_file_with_speakers(file_path: Path, model_name: str) -> None:
    """Process a single audio/video file and save its transcription with speakers and timestamps."""
    # Check if whisper-diarization is available before processing
    if not WHISPER_DIARIZATION_AVAILABLE:
        error_msg = """
ERROR: Cannot use --with-speakers option!

whisper-diarization dependencies are required for speaker diarization.
This should have been installed automatically. Please check the installation.

Use --with-timestamps instead for timestamps without speakers.
"""
        print(error_msg)
        raise SystemExit(1)
    
    try:
        # Use the integrated whisper-diarization pipeline
        # This does everything in one go: transcription + speaker identification
        transcription = transcribe_with_speaker_diarization(str(file_path), model_name)
        
        output_path = file_path.with_suffix('.txt')
        output_path.write_text(transcription, encoding="utf-8")
        print(f"Transcription with speakers and timestamps saved to: {output_path}")
        
    except SystemExit:
        raise  # Re-raise SystemExit to actually exit
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise SystemExit(1)

def process_file_with_assemblyai(file_path: Path, speaker_names: Optional[List[str]] = None) -> None:
    """Process a single audio/video file using AssemblyAI speaker diarization."""
    if not ASSEMBLYAI_AVAILABLE:
        error_msg = """
ERROR: Cannot use --use-assemblyai option!

AssemblyAI module is not available. Please check that:
1. assemblyai_diarize.py exists in the src directory
2. ASSEMBLYAI_API_KEY is set in your .env file
"""
        print(error_msg)
        raise SystemExit(1)
    
    try:
        print(f"Processing {file_path.name} with AssemblyAI...")
        transcription = transcribe_with_assemblyai(
            str(file_path),
            speaker_names=speaker_names,
            quiet=False
        )
        
        output_path = file_path.with_suffix('.txt')
        output_path.write_text(transcription, encoding="utf-8")
        print(f"Transcription with speakers saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise SystemExit(1)

def get_media_files(directory: Path) -> List[Path]:
    """Get all media files from the directory."""
    media_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.avi', '.mov'}
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in media_extensions]

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using OpenAI's Whisper or AssemblyAI")
    parser.add_argument("file", nargs="?", help="Path to a single audio/video file")
    parser.add_argument("--directory", help="Path to directory containing audio/video files")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: base)")
    parser.add_argument("--with-timestamps", action="store_true",
                      help="Include timestamps in the transcription output")
    parser.add_argument("--with-speakers", action="store_true",
                      help="Include speaker identification and timestamps (requires pyannote-audio)")
    parser.add_argument("--use-assemblyai", action="store_true",
                      help="Use AssemblyAI for transcription with speaker diarization")
    parser.add_argument("--speaker-names", type=str,
                      help="Comma-separated list of speaker names (e.g., 'Alice,Bob'). Only used with --use-assemblyai")
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.error("Either file path or --directory must be specified")
    
    # Parse speaker names if provided
    speaker_names = None
    if args.speaker_names:
        speaker_names = [name.strip() for name in args.speaker_names.split(',')]
        if not args.use_assemblyai:
            print("Warning: --speaker-names is only used with --use-assemblyai")
    
    try:
        if args.file:
            # Process single file
            file_path = Path(args.file)
            if args.use_assemblyai:
                process_file_with_assemblyai(file_path, speaker_names)
            elif getattr(args, 'with_speakers', False):
                process_file_with_speakers(file_path, args.model)
            elif getattr(args, 'with_timestamps', False):
                process_file_with_timestamps(file_path, args.model)
            else:
                process_file(file_path, args.model)
        else:
            # Process directory
            directory = Path(args.directory)
            if not directory.exists() or not directory.is_dir():
                raise NotADirectoryError(f"Directory not found: {directory}")
            
            media_files = get_media_files(directory)
            if not media_files:
                print(f"No media files found in {directory}")
                return 0
            
            print(f"Found {len(media_files)} media files to process")
            for file_path in media_files:
                if args.use_assemblyai:
                    process_file_with_assemblyai(file_path, speaker_names)
                elif getattr(args, 'with_speakers', False):
                    process_file_with_speakers(file_path, args.model)
                elif getattr(args, 'with_timestamps', False):
                    process_file_with_timestamps(file_path, args.model)
                else:
                    process_file(file_path, args.model)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 