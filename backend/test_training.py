#!/usr/bin/env python3
"""
Quick test script to validate training setup
Run this before full training to catch issues early
"""

import os
import sys
import torch
from pathlib import Path

def check_environment():
    """Check if environment is ready for training"""
    print("🔍 Checking training environment...")
    
    issues = []
    
    # Check directory structure
    required_dirs = ['data/real', 'data/fake', 'models', 'logs', 'train']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Check data
    real_images = len([f for f in os.listdir('data/real') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('data/real') else 0
    fake_images = len([f for f in os.listdir('data/fake') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('data/fake') else 0
    
    print(f"📊 Dataset: {real_images} real images, {fake_images} fake images")
    
    if real_images == 0:
        issues.append("No real images found in data/real/")
    if fake_images == 0:
        issues.append("No fake images found in data/fake/")
    if real_images < 50 or fake_images < 50:
        print("⚠️  Warning: Small dataset detected. Consider adding more images for better performance.")
    
    # Check Python environment
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check config file
    if not os.path.exists('config.yaml'):
        issues.append("Missing config.yaml file")
    
    # Check training scripts
    if not os.path.exists('train/train_model.py'):
        issues.append("Missing train/train_model.py")
    if not os.path.exists('train/utils.py'):
        issues.append("Missing train/utils.py")
    
    return issues

def test_data_loading():
    """Test if data loading works"""
    print("\n🔄 Testing data loading...")
    
    try:
        sys.path.append('train')
        from train_model import TrainingConfig, DataPipeline
        
        # Load config
        import yaml
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                config_dict = yaml.safe_load(f)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
        
        # Test data pipeline
        data_pipeline = DataPipeline(config)
        train_loader, val_loader, test_loader, class_names = data_pipeline.create_datasets()
        
        print(f"✅ Data loading successful!")
        print(f"   Classes: {class_names}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test one batch
        for batch_x, batch_y in train_loader:
            print(f"   Batch shape: {batch_x.shape}")
            print(f"   Labels shape: {batch_y.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test if model can be created"""
    print("\n🤖 Testing model creation...")
    
    try:
        sys.path.append('train')
        from train_model import ProductionDeepfakeDetector, TrainingConfig
        
        config = TrainingConfig()
        model = ProductionDeepfakeDetector(config)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model creation successful!")
        print(f"   Model: {config.backbone}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    print("🧪 Training Environment Test")
    print("=" * 50)
    
    # Check environment
    issues = check_environment()
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before training.")
        return False
    
    print("✅ Environment check passed!")
    
    # Test data loading
    if not test_data_loading():
        return False
    
    # Test model creation
    if not test_model_creation():
        return False
    
    print("\n🎉 All tests passed! Ready for training.")
    print("\nTo start training, run:")
    print("cd backend && python train/train_model.py --config config.yaml")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)