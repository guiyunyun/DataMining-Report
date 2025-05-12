import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertModel,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)
from dataset.feature_engineering import FeatureExtractor

class DepressionDataset(Dataset):
    """
    PyTorch dataset for depression detection
    """
    def __init__(self, texts, labels, tokenizer, max_length=128, 
                 include_features=False, feature_extractor=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_features = include_features
        self.feature_extractor = feature_extractor
        
        # Preprocess texts and extract features
        if include_features and feature_extractor:
            self.features = feature_extractor.extract_features(texts)
        else:
            self.features = None
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Use tokenizer to encode the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Build return dictionary
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Add engineered features
        if self.include_features and self.features is not None:
            features = self.features.iloc[idx].values
            item['features'] = torch.tensor(features, dtype=torch.float)
            
        return item

class DistilBERTBaselineModel(nn.Module):
    """
    DistilBERT-based baseline model
    """
    def __init__(self, num_classes=3):
        super(DistilBERTBaselineModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class DistilBERTWithFeaturesModel(nn.Module):
    """
    DistilBERT model with engineered features
    """
    def __init__(self, num_classes=3, num_features=5):
        super(DistilBERTWithFeaturesModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Dimension of engineered features (VADER sentiment scores, 4 features + pronoun usage frequency, 1 feature)
        self.num_features = num_features
        
        # Define a fusion layer to combine BERT representation and features
        self.fusion = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size + self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification layer
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, input_ids, attention_mask, features):
        # Get BERT output
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Concatenate BERT output and features
        combined = torch.cat((pooled_output, features), dim=1)
        
        # Through fusion layer
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits

class PilotModelTrainer:
    """
    Pilot model trainer
    """
    def __init__(self, model, tokenizer, device, feature_extractor=None, include_features=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.feature_extractor = feature_extractor
        self.include_features = include_features
        self.model.to(self.device)
        
    def train(self, train_df, val_df, batch_size=16, epochs=3, learning_rate=2e-5):
        """
        Train the model
        
        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            batch_size: Batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Create datasets
        train_dataset = DepressionDataset(
            train_df['text'].values, 
            train_df['labels'].values, 
            self.tokenizer,
            include_features=self.include_features,
            feature_extractor=self.feature_extractor
        )
        
        val_dataset = DepressionDataset(
            val_df['text'].values, 
            val_df['labels'].values, 
            self.tokenizer,
            include_features=self.include_features,
            feature_extractor=self.feature_extractor
        )
        
        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_f1 = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                if self.include_features:
                    features = batch['features'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, features)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            print(f"Validation Metrics: {val_metrics}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_model_state = self.model.state_dict().copy()
                
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return self.model
    
    def evaluate(self, dataloader):
        """
        Evaluate model performance
        
        Args:
            dataloader: Data loader
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.include_features:
                    features = batch['features'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, features)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Calculate ROC-AUC (multi-class using one-vs-rest method)
        all_probs = np.array(all_probs)
        try:
            roc_auc = roc_auc_score(
                np.eye(len(np.unique(all_labels)))[all_labels], 
                all_probs, 
                multi_class='ovr'
            )
        except ValueError:
            # May fail if some classes don't appear in the data
            roc_auc = 0.0
            
        # Create confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'macro_f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, test_df):
        """
        Make predictions on test data
        
        Args:
            test_df: Test data DataFrame
            
        Returns:
            predictions: Prediction results
            metrics: Evaluation metrics (if test_df contains labels)
        """
        # Create test dataset
        test_dataset = DepressionDataset(
            test_df['text'].values, 
            test_df['labels'].values if 'labels' in test_df.columns else np.zeros(len(test_df)),
            self.tokenizer,
            include_features=self.include_features,
            feature_extractor=self.feature_extractor
        )
        
        # Create data loader
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Calculate metrics if labels are available
        if 'labels' in test_df.columns:
            metrics = self.evaluate(test_dataloader)
        else:
            metrics = None
            
        # Predict
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                if self.include_features:
                    features = batch['features'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, features)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return all_preds, all_probs, metrics
    
    def save_model(self, output_dir):
        """
        Save the model
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Model path
        """
        self.model.load_state_dict(torch.load(model_path))
        return self.model 