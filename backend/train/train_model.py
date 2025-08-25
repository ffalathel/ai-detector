"""
Production-Ready Deepfake Detection System
Industry-standard implementation with comprehensive features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import tempfile
from PIL import Image
import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
import random
import time
import warnings
import wandb
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    set_seed,
    get_device,
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    get_lr_scheduler,
    MixedPrecisionTrainer,
    WarmupScheduler,
)

# Metrics and Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)

from tqdm import tqdm

# Optional MLOps and Monitoring (with fallbacks)
try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Experiment tracking disabled.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. TensorBoard logging disabled.")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not available. MLflow tracking disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")

# Model Serving and API (optional) - DISABLED to prevent conflicts with main app
FASTAPI_AVAILABLE = False
print("Warning: FastAPI serving disabled in training mode to prevent conflicts with main app")

# Metrics tracker
from metrics_tracker import MetricsTracker

# Configuration Management
@dataclass
class TrainingConfig:
    # Model Configuration
    backbone: str = 'resnet50'
    pretrained: bool = True
    dropout: float = 0.5
    num_classes: int = 2
    input_size: List[int] = None  # [C, H, W]
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    
    # Data Configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    
    # Regularization
    patience: int = 15
    min_delta: float = 1e-4
    grad_clip_value: float = 1.0
    
    # Paths
    data_path: str = "data"
    output_dir: str = "models"
    log_dir: str = "logs"
    
    # Monitoring
    use_wandb: bool = True
    use_tensorboard: bool = True
    use_mlflow: bool = True
    
    # Model Ensemble
    use_ensemble: bool = False
    ensemble_models: List[str] = None
    
    # Production
    model_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    IMAGENET_MEAN: List[float] = None
    IMAGENET_STD: List[float] = None

    
    resume: bool = False                # whether to resume at all
    resume_from: Optional[str] = None   # path to checkpoint (if None, pick default)
    resume_best: bool = True 
    use_mixed_precision: bool = True
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['resnet50', 'efficientnet_b0', 'efficientnet_b3']
        if self.input_size is None:
            self.input_size = [3, 224, 224]
        if self.IMAGENET_MEAN is None:
            self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        if self.IMAGENET_STD is None:
            self.IMAGENET_STD = [0.229, 0.224, 0.225]
    
     

# Enhanced Logging Setup
def setup_logging(config: TrainingConfig) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('deepfake_detector')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger

# Reproducibility Manager
class ReproducibilityManager:
    @staticmethod
    def set_seed(seed: int = 42):
        """Set seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def extract_labels_from_dataloader(loader: DataLoader) -> List[int]:
    """
    Robustly extract labels from a DataLoader's underlying dataset.
    Falls back to iterating the base dataset if necessary (costly).
    """
    dataset = loader.dataset
    # Unwrap Subset -> dataset.dataset
    try:
        # If it's a Subset, get the underlying dataset and indices
        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base = dataset.dataset
            indices = dataset.indices
            # prefer base.targets or base.labels if available
            targets = getattr(base, "targets", None) or getattr(base, "labels", None)
            if targets is not None:
                # handle torch tensors or lists
                return [int(targets[i]) for i in indices]
            # fallback: iterate through subset
            labels = []
            for i in indices:
                item = base[i]
                # item can be (x, y) or dict
                if isinstance(item, (tuple, list)):
                    labels.append(int(item[1]))
                elif isinstance(item, dict) and "label" in item:
                    labels.append(int(item["label"]))
                else:
                    raise ValueError("Cannot extract label from dataset item.")
            return labels
        else:
            base = getattr(dataset, "dataset", dataset)
            targets = getattr(base, "targets", None) or getattr(base, "labels", None)
            if targets is not None:
                return [int(t) for t in targets]
            # Final fallback: iterate (may be slow)
            labels = []
            for _, y in base:
                labels.append(int(y))
            return labels
    except Exception as e:
        logging.getLogger(f"Failed to extract labels robustly: {e}. Falling back to iteration.")
        labels = []
        for _, y in loader.dataset:
            labels.append(int(y))
        return labels

# Top-level dataset wrapper to ensure picklability when using multiple workers
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.samples = [self.base_dataset.samples[i] for i in indices]
    
    def __getitem__(self, idx):
        img_path, label = self.base_dataset.samples[self.indices[idx]]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.indices)

# Advanced Data Pipeline
class DataPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        self.test_transform = self._get_test_transforms()
    
    def _get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
            transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
        ])
    
    def _get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.IMAGENET_MEAN, std=self.config.IMAGENET_STD)
        ])
    
    def _get_test_transforms(self):
        """Test-time augmentation transforms"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.FiveCrop((224, 224)),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)
                ) for crop in crops
            ]))
        ])
    
    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """Create train, validation, and test data loaders"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.config.data_path}")
        
        # Create base dataset WITHOUT transforms
        base_dataset = datasets.ImageFolder(root=self.config.data_path, transform=None)
        
        if len(base_dataset) == 0:
            raise ValueError(f"No images found in {self.config.data_path}")
        class_counts = Counter([label for _, label in base_dataset.samples])
        if len(class_counts) < 2:
            raise ValueError(f"Dataset must contain at least 2 classes, found: {list(class_counts.keys())}")
    
        if min(class_counts.values()) < 10:  # Minimum samples per class
            raise ValueError(f"Each class must have at least 10 samples. Current distribution: {dict(class_counts)}")
        
        valid_samples = []
        corrupted_count = 0
        for img_path, label in base_dataset.samples:
            try:
                # Test if image can be opened and converted
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                with Image.open(img_path) as img:
                    img.convert('RGB')
                valid_samples.append((img_path, label))
            except Exception as e:
                corrupted_count += 1
                if corrupted_count <= 10:
                    logging.getLogger(__name__).warning(f"Corrupted image {img_path}: {e}")
                elif corrupted_count == 11:
                    logging.getLogger(__name__).warning("Additional corrupted images will not be logged individually")
    
        if corrupted_count > 0:
            logging.getLogger(__name__).warning(f"Found {corrupted_count} corrupted images out of {len(base_dataset.samples)} total")
        
        if len(valid_samples) < 20:  # Minimum total valid samples
            raise ValueError(f"Too many corrupted images. Only {len(valid_samples)} valid images found.")
    
        base_dataset.samples = valid_samples
        base_dataset.targets = [label for _, label in valid_samples]

        # Stratified split
        train_indices, val_indices, test_indices = self._stratified_split(base_dataset)
        
        # Use top-level TransformDataset (picklable) instead of nested class
        train_dataset = TransformDataset(base_dataset, train_indices, self.train_transform)
        val_dataset = TransformDataset(base_dataset, val_indices, self.val_transform)
        test_dataset = TransformDataset(base_dataset, test_indices, self.val_transform)
        
        # Weighted sampling for imbalanced data
        train_sampler = self._create_weighted_sampler(base_dataset, train_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=False,  # EXPLICIT: Required when using custom sampler
            num_workers=min(self.config.num_workers, 4),  # Limit workers
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers= False,  
            drop_last=True  # Prevent batch size inconsistencies
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0
        )
    
        return train_loader, val_loader, test_loader, base_dataset.classes
    
    def _stratified_split(self, dataset, train_ratio=None, val_ratio=None, seed = 42):
        """Create stratified train/val/test split with proper seeding"""
        if seed is None:
            seed = getattr(self.config, "seed", 42)
        if train_ratio is None:
            train_ratio = self.config.train_ratio
        if val_ratio is None:
            val_ratio = self.config.val_ratio
            
        # Use dataset samples for labels
        labels = [label for _, label in dataset.samples]
        class_indices = {}
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)
        
        # FIXED: Use numpy for reproducible shuffling
        np.random.seed(seed)
        train_indices, val_indices, test_indices = [], [], []
        
        for indices in class_indices.values():
            indices = np.array(indices)
            np.random.shuffle(indices)  # Seeded shuffle
            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)
            
            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train:n_train+n_val].tolist())
            test_indices.extend(indices[n_train+n_val:].tolist())
        
        # Final shuffle of the combined indices (also seeded)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        rng = np.random.default_rng(seed)
        
        return train_indices, val_indices, test_indices

    
    def _create_weighted_sampler(self, dataset, indices):
        """Create weighted sampler for imbalanced datasets"""
        labels = [dataset.samples[i][1] for i in indices]
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        return WeightedRandomSampler(weights, len(weights))

# Production-Ready Model Architecture
class ProductionDeepfakeDetector(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.backbone_name = config.backbone
        
        # Initialize backbone
        self.backbone = self._create_backbone()
        self.feature_dim = self._get_feature_dim()
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.25),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self):
        """Create backbone network"""
        if self.backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if self.config.pretrained else None)
            backbone.fc = nn.Identity()
        elif self.backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.config.pretrained else None)
            backbone.classifier = nn.Identity()
        elif self.backbone_name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if self.config.pretrained else None)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return backbone
    
    def _get_feature_dim(self):
        """Get feature dimension of backbone"""
        if 'resnet' in self.backbone_name:
            return 2048
        elif 'efficientnet_b0' in self.backbone_name:
            return 1280
        elif 'efficientnet_b3' in self.backbone_name:
            return 1536
        else:
            return 2048
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
    
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits

# Model Ensemble
class ModelEnsemble:
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)
    
    def predict(self, x):
        """Ensemble prediction"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
        
        # Weighted average
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return ensemble_pred

# Advanced Training Manager
class TrainingManager:
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = get_device()
        
        
        # Initialize monitoring
        self.setup_monitoring()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.use_amp = torch.cuda.is_available() and getattr(self.config, 'use_mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'learning_rates': [], 'val_f1': [], 'val_auc': []
        }
    
    def setup_monitoring(self):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitoring tools with availability checks
        self.tb_writer = None
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tb_writer = SummaryWriter(log_dir=f"{self.config.log_dir}/tensorboard")
                self.logger.info("TensorBoard initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="deepfake-detection",
                    config=asdict(self.config),
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.logger.info("Wandb initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Wandb: {e}")
        
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment("deepfake_detection")
                mlflow.start_run()
                self.logger.info("MLflow initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MLflow: {e}")
    
    def setup_model_and_training(self, num_classes: int, class_weights: torch.Tensor):
        # Model
        self.model = ProductionDeepfakeDetector(self.config).to(self.device)
        
        if num_classes == 2:
            # For binary classification with BCEWithLogitsLoss
            # pos_weight should be the ratio of negative to positive samples
            n_neg = float(class_weights[0])
            n_pos = float(class_weights[1])
        
            if n_pos <= 0:
                self.logger.warning("No positive samples in the training set; using pos_weight=1.0")
                pos_weight = torch.tensor(1.0, device=self.device)
            elif n_neg <= 0:
                self.logger.warning("No negative samples in the training set; using pos_weight=1.0")
                pos_weight = torch.tensor(1.0, device=self.device)
            else:
                pos_weight = torch.tensor(n_neg / n_pos, device=self.device)
        else:
            pos_weight = torch.tensor(1.0, device=self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer with different learning rates
        backbone_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Warmup scheduler
        self.warmup_scheduler = WarmupScheduler(
            self.optimizer, self.config.warmup_epochs, self.config.learning_rate
        )

    
    def train_epoch(self, train_loader):
        train_loss, train_metrics, _, _, _  = train_one_epoch(
            model=self.model,
            dataloader=train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.criterion,
            scaler=self.scaler,
            clip_grad=self.config.grad_clip_value,
            warmup_scheduler=self.warmup_scheduler,
            current_epoch=self.current_epoch
        )
        return train_loss, train_metrics['accuracy']
    
    def validate(self, val_loader):
        val_loss, metrics, labels, preds, scores = validate_one_epoch(
            model=self.model,
            dataloader=val_loader,
            device=self.device,
            criterion=self.criterion
        )
        return val_loss, metrics, labels, preds, scores
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):    
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.current_epoch,
            metrics=metrics,
            output_dir=Path(self.config.output_dir),
            is_best=is_best,
            logger=self.logger
        )
    def resume_training(self):
        """Resume training from a saved checkpoint if config.resume is set."""
        if not self.config.resume:
            return  # nothing to do

        checkpoint_path = Path(self.config.resume_from) if self.config.resume_from \
            else Path(self.config.output_dir) / "best_model.pt"

        if not checkpoint_path.exists():
            if self.logger:
                self.logger.warning(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return
        try:
            last_epoch, last_metrics = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                logger=self.logger,
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )

            self.current_epoch = last_epoch + 1
            if last_metrics:
                self.best_metrics = last_metrics

            if self.logger:
                self.logger.info(f"Resuming training from epoch {self.current_epoch}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to resume training: {e}")
        
    
    def log_metrics(self, train_loss, train_acc, val_loss, val_metrics):
        try:
            self.logger.info(f"Epoch [{self.current_epoch+1}] - "
                             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                             f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics.get('f1', 0):.4f}, "
                             f"Val AUC: {val_metrics.get('auc', 0):.4f}")
            
            if self.config.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
                self.tb_writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
                self.tb_writer.add_scalar('Accuracy/Train', train_acc, self.current_epoch)
                self.tb_writer.add_scalar('Accuracy/Validation', val_metrics.get('accuracy', 0), self.current_epoch)
                self.tb_writer.add_scalar('F1/Validation', val_metrics.get('f1', 0), self.current_epoch)
                self.tb_writer.add_scalar('AUC/Validation', val_metrics.get('auc', 0), self.current_epoch)
                self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
            if self.config.use_wandb and WANDB_AVAILABLE:
                try:
                    wandb.log({
                        'epoch': self.current_epoch,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_metrics.get('accuracy', 0),
                        'val_f1': val_metrics.get('f1', 0),
                        'val_auc': val_metrics.get('auc', 0),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                except Exception as e:
                    self.logger.warning(f"Wandb logging failed: {e}")
            
            if self.config.use_mlflow and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_f1': val_metrics.get('f1', 0),
                        'val_auc': val_metrics.get('auc', 0)
                    }, step=self.current_epoch)
                except Exception as e:
                    self.logger.warning(f"MLflow logging failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error during metric logging at epoch {self.current_epoch}: {e}", exc_info=True)

    def cleanup_monitoring(self):
        """Clean up monitoring resources"""
        try:
            if self.tb_writer is not None:
                self.tb_writer.close()
        except Exception as e:
            self.logger.error(f"Error closing TensorBoard writer: {e}")
        
        try:
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
        except Exception as e:
            self.logger.error(f"Error finishing Wandb: {e}")
        
        try:
            if self.config.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()
        except Exception as e:
            self.logger.error(f"Error ending MLflow run: {e}")

# FastAPI Model Serving - DISABLED to prevent conflicts with main app
# Use the main FastAPI app in app/main.py instead
class ModelServer:
    def __init__(self, model_path: str, config: TrainingConfig):
        raise NotImplementedError("ModelServer disabled. Use the main FastAPI app in app/main.py for serving")
    
    def setup_routes(self):
        pass
    
    def start_server(self):
        pass

def generate_evaluation_report(y_true, y_pred, y_scores, class_names, config, history):
    """Generate comprehensive evaluation report with plots and JSON output"""
    try:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Only calculate ROC if we have both classes
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
        else:
            roc_auc = 0.0
            pr_auc = 0.0
        
        # Plots (only if matplotlib available)
        if PLOTTING_AVAILABLE:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_xlabel('Predicted')
            axes[0,0].set_ylabel('Actual')
            
            # Training History
            axes[0,1].plot(history['train_loss'], label='Train Loss')
            axes[0,1].plot(history['val_loss'], label='Val Loss')
            axes[0,1].set_title('Training Loss')
            axes[0,1].legend()
            
            axes[0,2].plot(history['train_acc'], label='Train Acc')
            axes[0,2].plot(history['val_acc'], label='Val Acc')
            axes[0,2].set_title('Training Accuracy')
            axes[0,2].legend()
            
            # ROC Curve
            if len(np.unique(y_true)) > 1:
                axes[1,0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                axes[1,0].plot([0, 1], [0, 1], 'k--')
                axes[1,0].set_title('ROC Curve')
                axes[1,0].set_xlabel('False Positive Rate')
                axes[1,0].set_ylabel('True Positive Rate')
                axes[1,0].legend()
            
            # Precision-Recall Curve
            if len(np.unique(y_true)) > 1:
                axes[1,1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
                axes[1,1].set_title('Precision-Recall Curve')
                axes[1,1].set_xlabel('Recall')
                axes[1,1].set_ylabel('Precision')
                axes[1,1].legend()
            
            # Learning Rate Schedule
            axes[1,2].plot(history['learning_rates'])
            axes[1,2].set_title('Learning Rate Schedule')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Learning Rate')
            
            plt.tight_layout()
            fig.savefig(output_dir / 'evaluation_report.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # JSON Report
        detailed_report = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'training_history': history,
            'config': asdict(config),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'detailed_evaluation_report.json', 'w') as f:
            json.dump(detailed_report, f, indent=2)
            
        return detailed_report
        
    except Exception as e:
        print(f"Error generating evaluation report: {e}")
        return None

def save_production_model(model, config, metrics):
    """Save model with metadata + ONNX"""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'model_version': config.model_version,
        'backbone': config.backbone,
        'input_size': config.input_size,
        'num_classes': config.num_classes,
        'metrics': metrics,
        'config': asdict(config),
        'created_at': datetime.now().isoformat(),
        'framework': 'pytorch',
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'size': config.input_size[1:]
        }
    }
    
    # Save PyTorch checkpoint
    production_checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    torch.save(production_checkpoint, output_dir / 'production_model.pt')
    
    with open(output_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Export ONNX
    try:
        model.eval()
        dummy_input = torch.randn(1, *metadata['input_size']).to(next(model.parameters()).device)
        torch.onnx.export(
            model, dummy_input, output_dir / 'production_model.onnx',
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("ONNX model exported successfully")
    except Exception as e:
        print(f"ONNX export failed: {e}")

# Main Training Function
def main():
    parser = argparse.ArgumentParser(description='Production Deepfake Detection Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    # Server mode disabled to prevent conflicts with main FastAPI app
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = TrainingConfig(**config_dict)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config from {args.config}: {e}")
            print("Using default configuration")
            config = TrainingConfig()
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = TrainingConfig()
        # Save default config
        try:
            with open(args.config, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
            print(f"Saved default configuration to {args.config}")
        except Exception as e:
            print(f"Failed to save default config: {e}")
    
    # Set seed for reproducibility
    set_seed(getattr(config, "seed", 42))
    
    # Validate configuration
    from utils import validate_config, log_system_info
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    # Log system information
    log_system_info()
    
    # Model serving mode - DISABLED to prevent conflicts with main app
    # Server mode removed to prevent conflicts with main FastAPI app
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting training with config: {config}")
    
    # Initialize trainer variable for cleanup
    trainer = None
    
    try:
        # Create data pipeline
        logger.info("Setting up data pipeline...")
        data_pipeline = DataPipeline(config)
        train_loader, val_loader, test_loader, class_names = data_pipeline.create_datasets()
        
        logger.info(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
        logger.info(f"Classes: {class_names}")
        
        # Calculate class weights using the fixed extraction method
        def extract_labels_from_dataset(dataset):
            """Extract labels from any dataset type"""
            if hasattr(dataset, 'indices') and hasattr(dataset, 'base_dataset'):
                # Custom TransformDataset
                return [dataset.base_dataset.samples[i][1] for i in dataset.indices]
            elif hasattr(dataset, 'samples'):
                # ImageFolder
                return [label for _, label in dataset.samples]
            elif hasattr(dataset, 'targets'):
                # Generic dataset with targets
                return list(dataset.targets)
            else:
                # Fallback: iterate through dataset
                labels = []
                for _, label in dataset:
                    labels.append(label)
                return labels
        
        labels = extract_labels_from_dataset(train_loader.dataset)
        class_counts = Counter(labels)
        # Guarantee both classes exist in the vector (even if count is 0)
        counts_vector = [class_counts.get(i, 0) for i in range(len(class_names))]
        class_weights = torch.tensor(counts_vector, dtype=torch.float)
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Initialize metrics tracker with context that affects accuracy
        tracker_context = {
            "model.backbone": config.backbone,
            "model.dropout": config.dropout,
            "training.batch_size": config.batch_size,
            "training.learning_rate": config.learning_rate,
            "training.weight_decay": config.weight_decay,
            "training.epochs": config.epochs,
            "data.train_ratio": config.train_ratio,
            "data.val_ratio": config.val_ratio,
            "data.num_workers": config.num_workers,
            "data.pin_memory": config.pin_memory,
            "seed": getattr(config, "seed", 42),
            "dataset.class_names": class_names,
            "dataset.class_counts": dict(class_counts),
        }
        metrics_tracker = MetricsTracker(output_dir=config.log_dir, context=tracker_context)
        
        # Initialize training manager
        logger.info("Initializing training manager...")
        trainer = TrainingManager(config, logger)
        trainer.setup_model_and_training(len(class_names), class_weights)
        
        logger.info(f"Model initialized: {sum(p.numel() for p in trainer.model.parameters())} parameters")
        logger.info(f"Training on device: {trainer.device}")
        
        # Resume training if requested
        logger.info("Checking for resume training...")
        trainer.resume_training()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.patience, 
            min_delta=config.min_delta,
            metric='f1'
        )
        
        # Training variables
        best_f1 = 0.0
        start_epoch = trainer.current_epoch
        
        # Training loop
        logger.info(f"Starting training from epoch {start_epoch}...")
        
        for epoch in range(start_epoch, config.epochs):
            trainer.current_epoch = epoch
            
            try:
                logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
                
                # Training phase
                train_loss, train_metrics, _, _, _ = trainer.train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_metrics, _, _,_ = trainer.validate(val_loader)
                
                # Update training history
                trainer.training_history['train_loss'].append(train_loss)
                trainer.training_history['val_loss'].append(val_loss)
                trainer.training_history['train_acc'].append(train_metrics['accuracy'])
                trainer.training_history['val_acc'].append(val_metrics['accuracy'])
                trainer.training_history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
                trainer.training_history['val_f1'].append(val_metrics['f1'])
                trainer.training_history['val_auc'].append(val_metrics['auc'])
                
                # Learning rate scheduling (only after warmup)
                if epoch >= config.warmup_epochs:
                    trainer.scheduler.step()
                
                # Logging metrics
                trainer.log_metrics(train_loss, train_metrics['accuracy'], val_loss, val_metrics)
                
                # Track metrics with context for analysis
                metrics_tracker.log_epoch(
                    epoch_index=epoch,
                    split="train",
                    metrics={"loss": float(train_loss), "accuracy": float(train_metrics['accuracy'])},
                    extra={"lr": float(trainer.optimizer.param_groups[0]['lr'])}
                )
                metrics_tracker.log_epoch(
                    epoch_index=epoch,
                    split="val",
                    metrics={
                        "loss": float(val_loss),
                        "accuracy": float(val_metrics.get('accuracy', 0.0)),
                        "f1": float(val_metrics.get('f1', 0.0)),
                        "auc": float(val_metrics.get('auc', 0.0)),
                        "precision": float(val_metrics.get('precision', 0.0)),
                        "recall": float(val_metrics.get('recall', 0.0)),
                    },
                    extra={"lr": float(trainer.optimizer.param_groups[0]['lr'])}
                )
                
                # Checkpoint saving
                is_best = val_metrics['f1'] > best_f1
                if is_best:
                    best_f1 = val_metrics['f1']
                    trainer.best_metrics = val_metrics.copy()
                    logger.info(f"New best model! F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
                if (epoch + 1) % 5 == 0:
                    periodic_metrics = {
                        "epoch": epoch,
                        "periodic": True,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_f1": val_metrics.get('f1', 0)
                    }
                    trainer.save_checkpoint(periodic_metrics, is_best=False)
                    logger.info(f"Periodic checkpoint saved at epoch {epoch + 1}")
                # Save checkpoint
                trainer.save_checkpoint(val_metrics, is_best)
                
                # Early stopping check
                if early_stopping(val_metrics, trainer.model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    logger.info(f"Best validation F1: {best_f1:.4f}")
                    early_stopping.load_best_weights(trainer.model)
                    break
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
                # Save emergency checkpoint
                emergency_metrics = {"epoch": epoch, "emergency": True, "error": str(e)}
                trainer.save_checkpoint(emergency_metrics, is_best=False)
                raise
        
        # Training completed successfully
        logger.info("Training completed successfully!")
        
        # Final evaluation on test set
        logger.info("Starting final evaluation on test set...")
        test_loss, test_metrics, test_labels, test_preds, test_scores = trainer.validate(test_loader)
        
        # Log final test metrics
        metrics_tracker.log_epoch(
            epoch_index=trainer.current_epoch,
            split="test",
            metrics={
                "loss": float(test_loss),
                "accuracy": float(test_metrics.get('accuracy', 0.0)),
                "f1": float(test_metrics.get('f1', 0.0)),
                "auc": float(test_metrics.get('auc', 0.0)),
                "precision": float(test_metrics.get('precision', 0.0)),
                "recall": float(test_metrics.get('recall', 0.0)),
            }
        )
        
        # Print final results
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Validation F1: {best_f1:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Test F1: {test_metrics.get('f1', 0):.4f}")
        logger.info(f"Test AUC: {test_metrics.get('auc', 0):.4f}")
        logger.info(f"Test Precision: {test_metrics.get('precision', 0):.4f}")
        logger.info(f"Test Recall: {test_metrics.get('recall', 0):.4f}")
        logger.info("=" * 60)
        
        # Generate comprehensive evaluation report
        logger.info("Generating evaluation report...")
        try:
            generate_evaluation_report(
                test_labels, test_preds, test_scores, 
                class_names, config, trainer.training_history
            )
            logger.info("Evaluation report generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate evaluation report: {e}")
        
        # Save final production model
        logger.info("Saving production model...")
        try:
            save_production_model(trainer.model, config, trainer.best_metrics)
            logger.info("Production model saved successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save production model: {e}")
        
        logger.info("All training tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user (Ctrl+C)")
        if trainer:
            logger.info("💾 Saving current state...")
            try:
                emergency_metrics = {"epoch": trainer.current_epoch, "interrupted": True}
                trainer.save_checkpoint(emergency_metrics, is_best=False)
                logger.info("Emergency checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        
        # Try to save emergency state
        if trainer:
            try:
                emergency_metrics = {
                    "epoch": getattr(trainer, 'current_epoch', 0), 
                    "error": str(e),
                    "emergency": True
                }
                trainer.save_checkpoint(emergency_metrics, is_best=False)
                logger.info("Emergency checkpoint saved despite error")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
        
        raise  # Re-raise the original exception
        
    finally:
        # Cleanup monitoring resources
        logger.info("🧹 Cleaning up resources...")
        if trainer:
            try:
                trainer.cleanup_monitoring()
                logger.info("Monitoring resources cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        logger.info("👋 Training session ended")


if __name__ == "__main__":
    main()