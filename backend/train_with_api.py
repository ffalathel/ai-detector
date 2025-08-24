#!/usr/bin/env python3
"""
Training script wrapper that sets TRAINING_MODE environment variable
to prevent conflicts with the FastAPI app.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n🛑 Training interrupted. Cleaning up...")
    # Reset environment variable
    os.environ["TRAINING_MODE"] = "false"
    sys.exit(0)

def main():
    """Main training function with proper environment setup"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting AI Detector Training in Training Mode")
    print("=" * 50)
    
    # Set training mode environment variable
    os.environ["TRAINING_MODE"] = "true"
    print("✅ TRAINING_MODE environment variable set to 'true'")
    print("📝 FastAPI app will run in training mode (models not loaded)")
    
    # Check if training script exists
    train_script = Path(__file__).parent / "train" / "train_model.py"
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        sys.exit(1)
    
    try:
        print(f"🎯 Running training script: {train_script}")
        print("=" * 50)
        
        # Run the training script
        result = subprocess.run([
            sys.executable, str(train_script)
        ], env=os.environ.copy())
        
        if result.returncode == 0:
            print("=" * 50)
            print("🎉 Training completed successfully!")
        else:
            print("=" * 50)
            print(f"❌ Training failed with exit code: {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
        sys.exit(1)
    finally:
        # Always reset environment variable
        os.environ["TRAINING_MODE"] = "false"
        print("✅ TRAINING_MODE reset to 'false'")
        print("🔄 FastAPI app can now load models normally")

if __name__ == "__main__":
    main()
