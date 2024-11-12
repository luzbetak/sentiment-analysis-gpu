import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze only the first few layers
        for param in list(self.bert.parameters())[:2]:
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        # Sentiment-specific attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Main classifier layers
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Multiple fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, n_classes)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier initialization."""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Get sequence output and pooled output
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output      # [batch_size, hidden_size]

        # Apply attention over sequence outputs
        attention_weights = self.attention(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)

        # Combine attended output with pooled output
        combined = self.layer_norm(attended_output + pooled_output)
        combined = self.dropout(combined)

        # First dense layer with residual connection
        residual = combined
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + residual

        # Second dense layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Final classification layer
        x = self.fc3(x)
        
        return x

    def get_attention_weights(self, input_ids, attention_mask):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            attention_weights = self.attention(sequence_output).squeeze(-1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            return attention_weights
