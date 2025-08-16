"""
Utility functions for deepfake detection training
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, average_precision_score
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the available device, GPU if possible."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
    logger: Optional[logging.Logger] = None
):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'is_best': is_best
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = output_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        if logger:
            logger.info(f"Best model saved at epoch {epoch} to {best_path}")
    
    if logger:
        logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, Dict[str, float]]:
    """Load checkpoint and return last epoch and metrics."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint.get('metrics', {})

def calculate_metrics(
    outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive metrics from model outputs and true labels."""
    # Convert to numpy
    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
    preds = (probs > threshold).astype(int)
    targets_np = targets.cpu().numpy().flatten().astype(int)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(targets_np, preds),
        'precision': precision_score(targets_np, preds, zero_division=0),
        'recall': recall_score(targets_np, preds, zero_division=0),
        'f1': f1_score(targets_np, preds, zero_division=0),
    }
    
    # AUC metrics (only if we have both classes)
    if len(np.unique(targets_np)) > 1:
        fpr, tpr, _ = roc_curve(targets_np, probs)
        metrics['auc'] = auc(fpr, tpr)
        metrics['avg_precision'] = average_precision_score(targets_np, probs)
    else:
        metrics['auc'] = 0.0
        metrics['avg_precision'] = 0.0
    
    return metrics

class EarlyStopping:
    """Early stopping with model checkpointing."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, metric: str = 'f1', mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        score = metrics.get(self.metric)
        if score is None:
            raise ValueError(f"Metric '{self.metric}' not found in metrics")
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save best model weights."""
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def load_best_weights(self, model: torch.nn.Module):
        """Load best model weights."""
        if self.best_weights:
            model.load_state_dict(self.best_weights)

def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0, norm_type: float = 2.0):
    """Clip gradients to avoid exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'StepLR',
    step_size: int = 5,
    gamma: float = 0.1,
    patience: int = 3,
    min_lr: float = 1e-6
):
    """Get learning rate scheduler."""
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=patience, min_lr=min_lr)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=step_size)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    return scheduler

class MixedPrecisionTrainer:
    """Mixed precision training helper."""
    def __init__(self, use_amp: bool = True):
        self.use_amp = use_amp
        if self.use_amp and not torch.cuda.is_available():
            self.use_amp = False
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def __enter__(self):
        return self.scaler

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class WarmupScheduler:
    """Learning rate warmup scheduler."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
    
    def step(self, epoch: float):
        """Update learning rate during warmup."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: torch.nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad: Optional[float] = None,
    warmup_scheduler: Optional[WarmupScheduler] = None,
    current_epoch: int = 0
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {current_epoch+1} Training")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device).float().unsqueeze(1)  # Ensure shape (B,1)

        # Warmup learning rate
        if warmup_scheduler and current_epoch < warmup_scheduler.warmup_epochs:
            warmup_scheduler.step(current_epoch + batch_idx / len(dataloader))

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad:
                clip_gradients(model, clip_grad)
            optimizer.step()

        running_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

    avg_loss = running_loss / len(dataloader)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs, all_targets)

    return avg_loss, metrics

def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Tuple[float, Dict[str, float], List, List, List]:
    """Run one validation epoch."""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            targets = targets.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_outputs.extend(preds)
            all_targets.extend(targets.cpu().numpy().flatten().astype(int))
            all_probs.extend(probs)

    avg_loss = running_loss / len(dataloader)
    
    # Calculate metrics using numpy arrays
    targets_np = np.array(all_targets)
    preds_np = np.array(all_outputs)
    probs_np = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(targets_np, preds_np),
        'precision': precision_score(targets_np, preds_np, zero_division=0),
        'recall': recall_score(targets_np, preds_np, zero_division=0),
        'f1': f1_score(targets_np, preds_np, zero_division=0),
    }
    
    # AUC metrics
    if len(np.unique(targets_np)) > 1:
        fpr, tpr, _ = roc_curve(targets_np, probs_np)
        metrics['auc'] = auc(fpr, tpr)
        metrics['avg_precision'] = average_precision_score(targets_np, probs_np)
    else:
        metrics['auc'] = 0.0
        metrics['avg_precision'] = 0.0

    return avg_loss, metrics, all_targets, all_outputs, all_probs

def create_directory_structure(base_path: str):
    """Create necessary directory structure for the project."""
    base_path = Path(base_path)
    directories = [
        'data/train/real',
        'data/train/fake',
        'data/val/real', 
        'data/val/fake',
        'data/test/real',
        'data/test/fake',
        'models',
        'logs',
        'logs/tensorboard'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_path}")

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def freeze_backbone(model: torch.nn.Module, freeze: bool = True):
    """Freeze or unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone parameters {status}")

def print_model_summary(model: torch.nn.Module):
    """Print a summary of the model."""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print("="*50 + "\n")

def setup_environment():
    """Setup environment variables and settings."""
    # Set memory allocation strategy for better GPU utilization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Set memory fraction to avoid OOM
        torch.cuda.empty_cache()
    
    # Set number of threads for CPU operations
    torch.set_num_threads(min(8, torch.get_num_threads()))
    
    print("Environment setup completed")

def validate_config(config) -> bool:
    """Validate configuration parameters."""
    errors = []
    
    # Check paths
    if not os.path.exists(config.data_path):
        errors.append(f"Data path does not exist: {config.data_path}")
    
    # Check numeric values
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.train_ratio + config.val_ratio >= 1.0:
        errors.append("Train ratio + validation ratio must be less than 1.0")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def log_system_info():
    """Log system information."""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"CPU count: {os.cpu_count()}")
    print("="*50 + "\n")