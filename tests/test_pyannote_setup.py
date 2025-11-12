#!/usr/bin/env python3
"""
Test script to verify pyannote-audio and Hugging Face token setup.
Run this to check if speaker diarization will work before processing large files.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import transcribe
sys.path.append(str(Path(__file__).parent.parent))

def test_pyannote_import():
    """Test if pyannote-audio can be imported."""
    print("🔍 Testing pyannote-audio import...")
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote-audio imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pyannote-audio: {e}")
        print("   Install with: pip install pyannote-audio")
        return False

def test_huggingface_token():
    """Test if Hugging Face token is configured."""
    print("\n🔍 Testing Hugging Face authentication...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get user info - this will fail if not authenticated
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Hugging Face authentication failed: {e}")
        print("   Run: huggingface-cli login")
        return False

def test_model_access():
    """Test if we can access the pyannote speaker diarization model."""
    print("\n🔍 Testing model access...")
    try:
        from pyannote.audio import Pipeline
        
        print("   Attempting to load pyannote/speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        if "gated" in str(e).lower() or "agreement" in str(e).lower():
            print("   You need to accept the user agreement at:")
            print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        elif "authentication" in str(e).lower() or "token" in str(e).lower():
            print("   Authentication issue. Run: huggingface-cli login")
        return False

def test_basic_functionality():
    """Test basic speaker diarization functionality with a dummy audio file."""
    print("\n🔍 Testing basic functionality...")
    try:
        from pyannote.audio import Pipeline
        import torch
        import numpy as np
        
        # Create a simple test audio (2 seconds of silence)
        sample_rate = 16000
        duration = 2.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # This should work even with silence
        print("   Running diarization on test audio...")
        diarization = pipeline({"waveform": torch.tensor(audio, dtype=torch.float32).unsqueeze(0), "sample_rate": sample_rate})
        
        print("✅ Basic functionality test passed!")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🧪 Testing pyannote-audio and Hugging Face setup\n")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_pyannote_import),
        ("Authentication Test", test_huggingface_token),
        ("Model Access Test", test_model_access),
        ("Functionality Test", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
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
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Speaker diarization should work correctly.")
        print("You can now use: python src/transcribe.py --with-speakers")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before using --with-speakers")
        
        if passed == 0:
            print("\n🔧 Quick setup guide:")
            print("1. pip install pyannote-audio")
            print("2. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Accept the user agreement")
            print("4. Run: huggingface-cli login")

if __name__ == "__main__":
    main()
