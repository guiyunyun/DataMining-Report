import os
import torch
import pandas as pd
import numpy as np
import argparse
from transformers import DistilBertTokenizer
from dataset.pilot_subset import get_pilot_data
from dataset.feature_engineering import FeatureExtractor
from models.pilot_models import (
    DistilBERTBaselineModel,
    DistilBERTWithFeaturesModel,
    PilotModelTrainer
)
from sklearn.metrics import classification_report

def create_data_directories():
    """Create necessary data directories"""
    os.makedirs("data/pilot_subset", exist_ok=True)
    os.makedirs("trained_models/pilot_baseline", exist_ok=True)
    os.makedirs("trained_models/pilot_modified", exist_ok=True)
    os.makedirs("results_quick", exist_ok=True)

def run_pilot_study():
    """
    Run a quick pilot study experiment with reduced parameters
    """
    # Create necessary directories
    create_data_directories()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    print("Loading data...")
    train_df = get_pilot_data("train")
    val_df = get_pilot_data("val")
    test_df = get_pilot_data("test")
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Load tokenizer
    print("加载 DistilBERT 分词器...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("分词器加载完成")
    
    # Process data and create feature extractor
    feature_extractor = FeatureExtractor()
    
    # 修改参数为所需值
    batch_size = 16
    epochs = 2
    learning_rate = 2e-5
    max_seq_length = 64  # 保持较合理的序列长度
    
    # Train and evaluate models
    # Baseline model
    print("\n" + "="*50)
    print("Training Baseline Model (DistilBERT only)")
    print("="*50)
    
    # Create model
    print("创建基线模型...")
    baseline_model = DistilBERTBaselineModel(num_classes=3)
    print("基线模型创建完成")
    
    # Create trainer
    trainer = PilotModelTrainer(
        model=baseline_model,
        tokenizer=tokenizer,
        device=device
    )
    
    # 增加训练样本量
    small_train_df = train_df.sample(n=min(300, len(train_df)), random_state=42) 
    small_val_df = val_df.sample(n=min(50, len(val_df)), random_state=42)
    
    # Train model
    print("Training baseline model...")
    trainer.train(
        train_df=small_train_df,
        val_df=small_val_df,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Save model
    trainer.save_model("trained_models/pilot_baseline")
    
    # Evaluate on test set
    print("Evaluating baseline model on test set...")
    small_test_df = test_df.sample(n=min(100, len(test_df)), random_state=42)
    _, _, baseline_metrics = trainer.predict(small_test_df)
    
    print("\nBaseline Model Test Results:")
    print(f"Macro F1: {baseline_metrics['macro_f1']:.2f}")
    print(f"Precision: {baseline_metrics['precision']:.2f}")
    print(f"Recall: {baseline_metrics['recall']:.2f}")
    print(f"ROC-AUC: {baseline_metrics['roc_auc']:.2f}")
    
    # Save results
    baseline_results = {
        "model": "Pilot Baseline (DistilBERT only)",
        "dataset": "Sample Dataset (Quick Pilot)",
        "macro_f1": baseline_metrics['macro_f1'],
        "precision": baseline_metrics['precision'],
        "recall": baseline_metrics['recall'],
        "roc_auc": baseline_metrics['roc_auc']
    }
    
    pd.DataFrame([baseline_results]).to_csv("results_quick/baseline_results.csv", index=False)
    
    # Modified model
    print("\n" + "="*50)
    print("Training Modified Model (DistilBERT + sentiment & pronoun features)")
    print("="*50)
    
    # Data preprocessing: add features
    print("Extracting features for train, validation and test sets...")
    small_train_df_with_features = feature_extractor.process_dataframe(small_train_df)
    small_val_df_with_features = feature_extractor.process_dataframe(small_val_df)
    small_test_df_with_features = feature_extractor.process_dataframe(small_test_df)
    
    # Create model
    print("创建改进模型...")
    modified_model = DistilBERTWithFeaturesModel(num_classes=3, num_features=5)
    print("改进模型创建完成")
    
    # Create trainer
    trainer = PilotModelTrainer(
        model=modified_model,
        tokenizer=tokenizer,
        device=device,
        feature_extractor=feature_extractor,
        include_features=True
    )
    
    # Train model
    print("Training modified model...")
    trainer.train(
        train_df=small_train_df_with_features,
        val_df=small_val_df_with_features,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Save model
    trainer.save_model("trained_models/pilot_modified")
    
    # Evaluate on test set
    print("Evaluating modified model on test set...")
    _, _, modified_metrics = trainer.predict(small_test_df_with_features)
    
    print("\nModified Model Test Results:")
    print(f"Macro F1: {modified_metrics['macro_f1']:.2f}")
    print(f"Precision: {modified_metrics['precision']:.2f}")
    print(f"Recall: {modified_metrics['recall']:.2f}")
    print(f"ROC-AUC: {modified_metrics['roc_auc']:.2f}")
    
    # Save results
    modified_results = {
        "model": "Pilot Modified (DistilBERT + sentiment & pronoun features)",
        "dataset": "Sample Dataset (Quick Pilot)",
        "macro_f1": modified_metrics['macro_f1'],
        "precision": modified_metrics['precision'],
        "recall": modified_metrics['recall'],
        "roc_auc": modified_metrics['roc_auc']
    }
    
    pd.DataFrame([modified_results]).to_csv("results_quick/modified_results.csv", index=False)
    
    # Compare results
    reference_results = pd.DataFrame([{
        "model": "Reference Model (Theoretical)",
        "dataset": "Full Dataset",
        "macro_f1": 0.583,
        "precision": float('nan'),
        "recall": float('nan'),
        "roc_auc": float('nan')
    }])
    
    results = pd.concat([
        pd.DataFrame([baseline_results]), 
        pd.DataFrame([modified_results])
    ], ignore_index=True)
    
    all_results = pd.concat([reference_results, results], ignore_index=True)
    
    # Print comparison table
    print("\n" + "="*80)
    print("Comparison Results:")
    print("="*80)
    print(all_results.to_string(index=False))
    
    # Save comparison results
    all_results.to_csv("results_quick/comparison_results.csv", index=False)
    
    # Calculate improvement percentage
    baseline_f1 = baseline_results['macro_f1']
    modified_f1 = modified_results['macro_f1']
    
    improvement = modified_f1 - baseline_f1
    improvement_percentage = (improvement / baseline_f1) * 100
    
    print(f"\nModified model improved over Baseline model by {improvement:.4f} absolute points ({improvement_percentage:.2f}%) in Macro F1")
    
    # Return results for future use
    return all_results

if __name__ == "__main__":
    run_pilot_study() 