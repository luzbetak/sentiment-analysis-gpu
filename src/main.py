#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
from transformers import BertTokenizer

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG, DEVICE, SEED
from src.models.classifier import SentimentClassifier
from src.data.dataset import TextDataset
from src.utils.training import train_model
from src.utils.logging_config import setup_logger
from src.data_generation.generate_data import generate_balanced_dataset, get_test_examples

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def label_to_sentiment(label: int) -> str:
    sentiments = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return sentiments.get(label, "Unknown")

def main():
    # Setup
    logger = setup_logger()
    set_seed(SEED)
    logger.info(f"Using device: {DEVICE}")

    # Generate dataset
    text_data, labels = generate_balanced_dataset()

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        text_data, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    # Create datasets
    train_dataset = TextDataset(
        train_texts, 
        train_labels, 
        max_length=MODEL_CONFIG['max_length']
    )
    val_dataset = TextDataset(
        val_texts, 
        val_labels, 
        max_length=MODEL_CONFIG['max_length']
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=MODEL_CONFIG['batch_size']
    )

    # Initialize model
    model = SentimentClassifier(
        n_classes=MODEL_CONFIG['n_classes'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    ).to(DEVICE)

    # Train model
    logger.info("Starting training...")
    model = train_model(model, train_dataloader, val_dataloader, logger)

    # Test predictions
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_texts = get_test_examples()

    for text in test_texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MODEL_CONFIG['max_length'],
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
            prediction = torch.argmax(probabilities).item()

            logger.info(f"\nText: {text}")
            logger.info(f"Predicted Sentiment: {label_to_sentiment(prediction)}")
            logger.info("Sentiment Probabilities:")
            for i, prob in enumerate(probabilities.cpu().numpy()):
                logger.info(f"{label_to_sentiment(i)}: {prob:.4f}")

if __name__ == "__main__":
    main()
