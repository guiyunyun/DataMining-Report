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
    os.makedirs("results_pilot", exist_ok=True)

def run_pilot_study(args):
    """
    Run pilot study experiment
    
    Args:
        args: Command line arguments
    """
    # Create necessary directories
    create_data_directories()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Process data and create feature extractor (for modified model)
    feature_extractor = FeatureExtractor()
    
    # Train and evaluate different models
    if args.model == "baseline" or args.model == "all":
        run_baseline_model(train_df, val_df, test_df, tokenizer, device, args)
        
    if args.model == "modified" or args.model == "all":
        run_modified_model(train_df, val_df, test_df, tokenizer, feature_extractor, device, args)
    
    # Compare results
    if args.model == "all":
        compare_results()

def run_baseline_model(train_df, val_df, test_df, tokenizer, device, args):
    """
    Train and evaluate baseline model (DistilBERT only)
    """
    print("\n" + "="*50)
    print("Training Baseline Model (DistilBERT only)")
    print("="*50)
    
    # Create model
    baseline_model = DistilBERTBaselineModel(num_classes=3)
    
    # Create trainer
    trainer = PilotModelTrainer(
        model=baseline_model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Train model
    if not args.skip_training:
        print("Training baseline model...")
        trainer.train(
            train_df=train_df,
            val_df=val_df,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Save model
        trainer.save_model("trained_models/pilot_baseline")
    else:
        # Load existing model
        print("Loading pre-trained baseline model...")
        trainer.load_model("trained_models/pilot_baseline/model_weights.pth")
    
    # Evaluate on test set
    print("Evaluating baseline model on test set...")
    _, _, metrics = trainer.predict(test_df)
    
    print("\nBaseline Model Test Results:")
    print(f"Macro F1: {metrics['macro_f1']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2f}")
    
    # Save results
    results = {
        "model": "Pilot Baseline (DistilBERT only)",
        "dataset": "LT-EDI 2022 (Subset)",
        "macro_f1": metrics['macro_f1'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "roc_auc": metrics['roc_auc']
    }
    
    pd.DataFrame([results]).to_csv("trained_models/pilot_baseline/results.csv", index=False)
    pd.DataFrame([results]).to_csv("results_pilot/baseline_results.csv", index=False)

def run_modified_model(train_df, val_df, test_df, tokenizer, feature_extractor, device, args):
    """
    Train and evaluate modified model (DistilBERT + features)
    """
    print("\n" + "="*50)
    print("Training Modified Model (DistilBERT + sentiment & pronoun features)")
    print("="*50)
    
    # Data preprocessing: add features
    print("Extracting features for train, validation and test sets...")
    train_df_with_features = feature_extractor.process_dataframe(train_df)
    val_df_with_features = feature_extractor.process_dataframe(val_df)
    test_df_with_features = feature_extractor.process_dataframe(test_df)
    
    # Create model
    modified_model = DistilBERTWithFeaturesModel(num_classes=3, num_features=5)
    
    # Create trainer
    trainer = PilotModelTrainer(
        model=modified_model,
        tokenizer=tokenizer,
        device=device,
        feature_extractor=feature_extractor,
        include_features=True
    )
    
    # Train model
    if not args.skip_training:
        print("Training modified model...")
        trainer.train(
            train_df=train_df_with_features,
            val_df=val_df_with_features,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Save model
        trainer.save_model("trained_models/pilot_modified")
    else:
        # Load existing model
        print("Loading pre-trained modified model...")
        trainer.load_model("trained_models/pilot_modified/model_weights.pth")
    
    # Evaluate on test set
    print("Evaluating modified model on test set...")
    _, _, metrics = trainer.predict(test_df_with_features)
    
    print("\nModified Model Test Results:")
    print(f"Macro F1: {metrics['macro_f1']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2f}")
    
    # Save results
    results = {
        "model": "Pilot Modified (DistilBERT + sentiment & pronoun features)",
        "dataset": "LT-EDI 2022 (Subset)",
        "macro_f1": metrics['macro_f1'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "roc_auc": metrics['roc_auc']
    }
    
    pd.DataFrame([results]).to_csv("trained_models/pilot_modified/results.csv", index=False)
    pd.DataFrame([results]).to_csv("results_pilot/modified_results.csv", index=False)

def compare_results():
    """Compare results from different models"""
    try:
        # Load results
        baseline_results = pd.read_csv("results_pilot/baseline_results.csv")
        modified_results = pd.read_csv("results_pilot/modified_results.csv")
        
        # Merge results
        results = pd.concat([baseline_results, modified_results], ignore_index=True)
        
        # Add reference results
        reference_results = pd.DataFrame([{
            "model": "Poświata & Perełkiewicz (2022)",
            "dataset": "LT-EDI 2022 (Full)",
            "macro_f1": 0.583,
            "precision": float('nan'),
            "recall": float('nan'),
            "roc_auc": float('nan')
        }])
        
        all_results = pd.concat([reference_results, results], ignore_index=True)
        
        # Print comparison table
        print("\n" + "="*80)
        print("Comparison Results:")
        print("="*80)
        print(all_results.to_string(index=False))
        
        # Save comparison results
        all_results.to_csv("results_pilot/comparison_results.csv", index=False)
        
        # Calculate improvement percentage
        if len(baseline_results) > 0 and len(modified_results) > 0:
            baseline_f1 = baseline_results['macro_f1'].values[0]
            modified_f1 = modified_results['macro_f1'].values[0]
            
            improvement = modified_f1 - baseline_f1
            improvement_percentage = (improvement / baseline_f1) * 100
            
            print(f"\nModified model improved over Baseline model by {improvement:.4f} absolute points ({improvement_percentage:.2f}%) in Macro F1")
            
    except Exception as e:
        print(f"Could not compare results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run depression detection pilot study experiment")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="all", 
        choices=["baseline", "modified", "all"],
        help="Type of model to run: baseline, modified, or all"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Training batch size"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate"
    )
    
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="Do not use CUDA even if available"
    )
    
    parser.add_argument(
        "--skip_training", 
        action="store_true", 
        help="Skip training, just load already trained models for evaluation"
    )
    
    args = parser.parse_args()
    run_pilot_study(args) 