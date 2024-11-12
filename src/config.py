import torch

# Model Configuration
MODEL_CONFIG = {
    'max_length': 128,
    'n_classes': 3,
    'dropout_rate': 0.3,
    'batch_size': 16,
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 30,
    'patience': 8,
    # Lower learning rates for stability
    'learning_rate_bert': 1e-5,
    'learning_rate_classifier': 5e-5,
    # More balanced regularization
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    # More balanced class weights
    'class_weights': [1.2, 1.5, 1.2],
    'focal_loss_gamma': 2,
    'gradient_clip_value': 1.0,
    # Mixup settings
    'mixup_alpha': 0.2,
    # Loss weights
    'ce_weight': 0.6,
    'focal_weight': 0.4
}

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random Seed
SEED = 42
