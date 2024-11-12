import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from ..config import TRAINING_CONFIG, DEVICE

def mixup_data(x, y, alpha=0.2, is_input_ids=False):
    """Performs mixup on the input and target."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    if is_input_ids:
        mixed_x = x if lam >= 0.5 else x[index]
    else:
        mixed_x = lam * x + (1 - lam) * x[index]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def focal_loss(outputs, targets, gamma=2):
    """Compute focal loss for dealing with class imbalance."""
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return ((1 - pt) ** gamma * ce_loss).mean()

def train_model(model, train_dataloader, val_dataloader, logger):
    """Train the sentiment classifier model."""
    # Initialize optimizers
    bert_params = list(model.bert.parameters())
    classifier_params = list(model.parameters())[len(bert_params):]

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': TRAINING_CONFIG['learning_rate_bert']},
        {'params': classifier_params, 'lr': TRAINING_CONFIG['learning_rate_classifier']}
    ], weight_decay=TRAINING_CONFIG['weight_decay'])

    total_steps = len(train_dataloader) * TRAINING_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[TRAINING_CONFIG['learning_rate_bert'], TRAINING_CONFIG['learning_rate_classifier']],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    no_improve = 0
    best_model = None
    class_weights = torch.tensor(TRAINING_CONFIG['class_weights']).to(DEVICE)

    for epoch in range(TRAINING_CONFIG['num_epochs']):
        model.train()
        total_loss = 0
        predictions_list = []
        labels_list = []
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{TRAINING_CONFIG["num_epochs"]}')
        
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Use mixup with 50% probability
            use_mixup = np.random.random() < 0.5

            if use_mixup:
                mixed_input_ids, y_a, y_b, lam = mixup_data(
                    input_ids, labels, TRAINING_CONFIG['mixup_alpha'], is_input_ids=True
                )
                mixed_attention_mask, _, _, _ = mixup_data(
                    attention_mask, labels, TRAINING_CONFIG['mixup_alpha'], is_input_ids=False
                )
                
                outputs = model(mixed_input_ids, mixed_attention_mask)
                loss = mixup_criterion(
                    lambda p, y: F.cross_entropy(p, y, weight=class_weights, label_smoothing=TRAINING_CONFIG['label_smoothing']),
                    outputs, y_a, y_b, lam
                )
            else:
                outputs = model(input_ids, attention_mask)
                ce_loss = F.cross_entropy(
                    outputs, labels, 
                    weight=class_weights,
                    label_smoothing=TRAINING_CONFIG['label_smoothing']
                )
                f_loss = focal_loss(outputs, labels, gamma=TRAINING_CONFIG['focal_loss_gamma'])
                loss = 0.7 * ce_loss + 0.3 * f_loss

                # Collect predictions only for non-mixup batches
                predictions = torch.argmax(outputs, dim=1)
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

            # Add L1 regularization
            l1_reg = 0
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += 0.01 * l1_reg

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=TRAINING_CONFIG['gradient_clip_value']
            )
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, labels, weight=class_weights)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'Average Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Only print classification reports if we have predictions
        if len(predictions_list) > 0:
            logger.info('\nTraining Classification Report:')
            logger.info(classification_report(labels_list, predictions_list, digits=4, zero_division=0))
        
        logger.info('\nValidation Classification Report:')
        logger.info(classification_report(val_labels, val_predictions, digits=4, zero_division=0))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            best_model = model.state_dict().copy()
            logger.info("New best model saved!")
        else:
            no_improve += 1
            if no_improve >= TRAINING_CONFIG['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    if best_model is not None:
        model.load_state_dict(best_model)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model
