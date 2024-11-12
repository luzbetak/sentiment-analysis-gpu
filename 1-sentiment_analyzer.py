#!/usr/bin/env python
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import logging
import numpy as np
from tqdm import tqdm
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length=64):
        self.texts      = texts
        self.labels     = labels
        self.max_length = max_length
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens    = True,
            max_length            = self.max_length,
            return_token_type_ids = False,
            padding               = 'max_length',
            truncation            = True,
            return_attention_mask = True,
            return_tensors        = 'pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        accuracy    = (predictions == labels).float().mean()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item() * len(labels)
        total_samples += len(labels)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels.to(device))
            total_loss += loss.item()

            batch_predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy()
            predictions.extend(batch_predictions)
            actual_labels.extend(labels.numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actual_labels, predictions)
    return avg_loss, accuracy, predictions, actual_labels

def main():
    set_seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare dataset
    text_data = [
        # Positive examples
        "This movie is amazing, I loved it!",
        "Brilliant performance by the entire cast!",
        "A masterpiece of modern cinema!",
        "Absolutely fantastic film!",
        "One of the best movies I've ever seen!",
        "Great story and excellent direction!",
        "This film exceeded all expectations!",
        "A true work of art!",
        # Negative examples
        "Terrible movie, complete waste of time.",
        "I really hated every minute of it.",
        "Poor acting and weak plot.",
        "One of the worst films ever made.",
        "Absolutely disappointing!",
        "This movie was a complete disaster.",
        "Boring and predictable throughout.",
        "Would not recommend to anyone."
    ]
    
    labels = [1]*8 + [0]*8  # 8 positive and 8 negative examples

    # Create train/val split manually for small dataset
    train_size   = 12
    train_texts  = text_data[:train_size]
    train_labels = labels[:train_size]
    val_texts    = text_data[train_size:]
    val_labels   = labels[train_size:]

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset   = TextDataset(val_texts, val_labels)

    # Create dataloaders with small batch size
    batch_size       = 4  # Small batch size for small dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and training components
    model     = SentimentClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs  = 5
    total_steps = len(train_dataloader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, criterion, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_dataloader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

    # Test new examples
    test_texts = [
        "This was an absolutely amazing movie! A true masterpiece!",
        "I really hated this film, terrible acting and awful storyline."
    ]
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    for text in test_texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length            = 64,
            return_token_type_ids = False,
            padding               = 'max_length',
            truncation            = True,
            return_attention_mask = True,
            return_tensors        = 'pt'
        )
        
        input_ids      = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs     = model(input_ids, attention_mask)
            probability = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        sentiment = 'Positive' if probability > 0.5 else 'Negative'
        logger.info(f"\nText: {text}")
        logger.info(f"Sentiment: {sentiment} (probability: {probability:.4f})")

if __name__ == "__main__":
    main()
