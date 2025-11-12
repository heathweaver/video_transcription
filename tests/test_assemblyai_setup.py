#!/usr/bin/env python3
"""
Test script to verify AssemblyAI integration.
Run this to check if AssemblyAI API is configured correctly.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_env_variable():
    """Test if ASSEMBLYAI_API_KEY is set."""
    print("🔍 Testing environment variable...")
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ASSEMBLYAI_API_KEY')
    if api_key:
        print(f"✅ ASSEMBLYAI_API_KEY found (length: {len(api_key)})")
        return True
    else:
        print("❌ ASSEMBLYAI_API_KEY not found in .env file")
        print("   Add it to your .env file: ASSEMBLYAI_API_KEY=your_key_here")
        return False

def test_import():
    """Test if the assemblyai_diarize module can be imported."""
    print("\n🔍 Testing module import...")
    try:
        import assemblyai_diarize
        print("✅ assemblyai_diarize module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import assemblyai_diarize: {e}")
        return False

def test_api_connection():
    """Test if we can connect to AssemblyAI API."""
    print("\n🔍 Testing API connection...")
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not api_key:
            print("❌ Cannot test connection without API key")
            return False
        
        # Test with a simple API call to check authentication
        headers = {"authorization": api_key}
        response = requests.get(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers
        )
        
        if response.status_code in [200, 404]:  # 404 is ok, means auth worked but no transcript ID
            print("✅ API connection successful")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed - check your API key")
            return False
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_cli_integration():
    """Test if the CLI accepts the new arguments."""
    print("\n🔍 Testing CLI integration...")
    try:
        import subprocess
        
        # Use sys.executable to get the current Python interpreter
        python_exe = sys.executable
        
        # Test if --use-assemblyai flag is recognized
        result = subprocess.run(
            [python_exe, "src/transcribe.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            help_text = result.stdout
            if "--use-assemblyai" in help_text and "--speaker-names" in help_text:
                print("✅ CLI flags are properly integrated")
                return True
            else:
                print("❌ CLI flags not found in help text")
                print(f"   Help output:\n{help_text}")
                return False
        else:
            print(f"❌ Failed to run transcribe.py --help: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def test_with_sample_audio():
    """Test with AssemblyAI's sample audio file."""
    print("\n🔍 Testing with sample audio...")
    
    try:
        import assemblyai_diarize
        
        # Use AssemblyAI's public sample audio URL (short podcast clip with 2 speakers)
        # This is a small ~30 second sample they provide for testing
        sample_url = "https://storage.googleapis.com/aai-web-samples/espn-bears.m4a"
        
        print("   Using AssemblyAI sample audio (short podcast clip)")
        print("   Testing with direct URL (no upload needed)...")
        
        # We can test with the URL directly
        from assemblyai_diarize import start_transcription, poll_transcription, format_transcript_with_speakers
        
        transcript_id = start_transcription(
            sample_url,
            speaker_labels=True,
            quiet=False
        )
        
        result = poll_transcription(transcript_id, quiet=False)
        formatted = format_transcript_with_speakers(result, speaker_names=["Host", "Guest"])
        
        if formatted:
            print("✅ Transcription completed successfully!")
            print(f"   Preview (first 200 chars):\n   {formatted[:200]}...")
            return True
        else:
            print("❌ Transcription returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests and provide summary."""
    print("🧪 Testing AssemblyAI Integration\n")
    print("=" * 60)
    
    tests = [
        ("Environment Variable", test_env_variable),
        ("Module Import", test_import),
        ("API Connection", test_api_connection),
        ("CLI Integration", test_cli_integration),
        ("Sample Audio Test", test_with_sample_audio)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            # Stop if critical tests fail
            if result is False and test_name in ["Environment Variable", "Module Import"]:
                print(f"\n⚠️  Stopping tests - {test_name} must pass first")
                break
                
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"{status} - {test_name}")
    
    print(f"\nPassed: {passed}, Failed: {failed}, Skipped: {skipped}")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All critical tests passed! AssemblyAI integration is ready.")
        print("\nUsage example:")
        print("  python src/transcribe.py audio.mp3 --use-assemblyai --speaker-names 'Alice,Bob'")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        if not os.getenv('ASSEMBLYAI_API_KEY'):
            print("\n🔧 Quick setup:")
            print("1. Get an API key from: https://www.assemblyai.com/")
            print("2. Add to .env file: ASSEMBLYAI_API_KEY=your_key_here")

if __name__ == "__main__":
    main()