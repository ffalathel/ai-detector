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
    import wandb
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

# Model Serving and API (optional)
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from PIL import Image
    import io
    import base64
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Model serving disabled.")

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
        
        full_dataset = datasets.ImageFolder(root=self.config.data_path)
        
        if len(full_dataset) == 0:
            raise ValueError(f"No images found in {self.config.data_path}")
        
        # Stratified split
        train_indices, val_indices, test_indices = self._stratified_split(full_dataset)
        
        # Create datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        # Apply transforms
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.val_transform
        test_dataset.dataset.transform = self.val_transform
        
        # Weighted sampling for imbalanced data
        train_sampler = self._create_weighted_sampler(full_dataset, train_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers= False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, test_loader, full_dataset.classes
    
    def _stratified_split(self, dataset, train_ratio=None, val_ratio=None):
        """Create stratified train/val/test split"""
        if train_ratio is None:
            train_ratio = self.config.train_ratio
        if val_ratio is None:
            val_ratio = self.config.val_ratio
            
        labels = [label for _, label in dataset.samples]
        class_indices = {}
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)
        
        train_indices, val_indices, test_indices = [], [], []
        for indices in class_indices.values():
            random.shuffle(indices)
            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train+n_val])
            test_indices.extend(indices[n_train+n_val:])
        
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
        set_seed(42)
        
        # Initialize monitoring
        self.setup_monitoring()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = torch.cuda.amp.GradScaler()
        
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
        
        # Loss function with class weights
        pos_weight = class_weights[0] / class_weights[1] if len(class_weights) > 1 else None
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
        train_loss, train_metrics = train_one_epoch(
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

# FastAPI Model Serving
class ModelServer:
    def __init__(self, model_path: str, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logger
        self.logger = logging.getLogger('model_server')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Load model
        self.model = ProductionDeepfakeDetector(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize FastAPI
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="Deepfake Detection API", version=config.model_version)
            self.setup_routes()
        else:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_version": self.config.model_version}
        
        @self.app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            try:
                self.logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")

                if not file.content_type.startswith('image/'):
                    self.logger.warning(f"Rejected non-image file: {file.content_type}")
                    raise HTTPException(status_code=400, detail="File must be an image")

                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(image_tensor)
                    probability = torch.sigmoid(logits).item()

                is_fake = probability > 0.5
                confidence = probability if is_fake else 1 - probability

                self.logger.info(f"Prediction: {'fake' if is_fake else 'real'}, confidence: {confidence:.4f}")

                return {
                    "prediction": "fake" if is_fake else "real",
                    "confidence": float(confidence),
                    "probability": float(probability),
                    "model_version": self.config.model_version
                }
            except Exception as e:
                self.logger.error(f"Prediction error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")
    
    def start_server(self):
        uvicorn.run(self.app, host=self.config.api_host, port=self.config.api_port)

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
    parser.add_argument('--serve', action='store_true', help='Start model server')
    parser.add_argument('--model_path', type=str, help='Model path for serving')
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
    
    # Model serving mode
    if args.serve:
        if not args.model_path:
            raise ValueError("Model path required for serving mode")
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn python-multipart pillow")
        server = ModelServer(args.model_path, config)
        server.start_server()
        return
    
    # Setup reproducibility
    ReproducibilityManager.set_seed(42)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting training with config: {config}")
    
    try:
        # Create data pipeline
        data_pipeline = DataPipeline(config)
        train_loader, val_loader, test_loader, class_names = data_pipeline.create_datasets()
        
        logger.info(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
        logger.info(f"Classes: {class_names}")
        
        # Calculate class weights
        train_labels = [train_loader.dataset.dataset.samples[i][1] for i in train_loader.dataset.indices]
        class_counts = Counter(train_labels)
        class_weights = torch.tensor([class_counts[i] for i in range(len(class_names))], dtype=torch.float)
        
        # Initialize training manager
        trainer = TrainingManager(config, logger)
        trainer.setup_model_and_training(len(class_names), class_weights)
        
        logger.info(f"Model initialized: {sum(p.numel() for p in trainer.model.parameters())} parameters")
        logger.info(f"Training on device: {trainer.device}")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.patience, 
            min_delta=config.min_delta,
            metric='f1'
        )
        
        # Training loop
        logger.info("Starting training...")
        best_f1 = 0.0
        
        for epoch in range(config.epochs):
            trainer.current_epoch = epoch
            
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics, val_labels, val_preds, val_scores = trainer.validate(val_loader)
            
            # Update history
            trainer.training_history['train_loss'].append(train_loss)
            trainer.training_history['val_loss'].append(val_loss)
            trainer.training_history['train_acc'].append(train_acc)
            trainer.training_history['val_acc'].append(val_metrics['accuracy'])
            trainer.training_history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
            trainer.training_history['val_f1'].append(val_metrics['f1'])
            trainer.training_history['val_auc'].append(val_metrics['auc'])
            
            # Scheduler step
            if epoch >= config.warmup_epochs:
                trainer.scheduler.step()
            
            # Logging
            trainer.log_metrics(train_loss, train_acc, val_loss, val_metrics)
            
            # Checkpoint saving
            is_best = val_metrics['f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['f1']
                trainer.best_metrics = val_metrics
                logger.info(f"New best model saved! F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            trainer.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if early_stopping(val_metrics, trainer.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                early_stopping.load_best_weights(trainer.model)
                break
        
        # Final evaluation on test set
        logger.info("Final evaluation on test set...")
        test_loss, test_metrics, test_labels, test_preds, test_scores = trainer.validate(test_loader)
        
        logger.info("=" * 50)
        logger.info("FINAL RESULTS")
        logger.info("=" * 50)
        logger.info(f"Best Validation F1: {best_f1:.4f}")
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Generate comprehensive evaluation report
        generate_evaluation_report(
            test_labels, test_preds, test_scores, 
            class_names, config, trainer.training_history
        )
        
        # Save final model for production
        save_production_model(trainer.model, config, trainer.best_metrics)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup monitoring
        if 'trainer' in locals():
            trainer.cleanup_monitoring()

if __name__ == "__main__":
    main()