# Depression Detection Pilot Study

This project is an extension of the research by Poświata & Perełkiewicz (2022), aiming to verify the feasibility of data mining and text analytics for early depression risk detection from social media, and to test the effectiveness of adding engineered features to pre-trained language models.

## Project Overview

This experiment consists of two main parts:

1. **Baseline Model**: Using DistilBERT for a three-class classification task (not depressed, moderately depressed, severely depressed).
2. **Modified Model**: Combining DistilBERT representations with engineered features such as VADER sentiment scores and first-person singular pronoun usage frequency.

## Dataset

The experiment uses a subset of the LT-EDI 2022 shared task dataset (approximately 2,000 posts), created through stratified random sampling:
- Training set: 1,400 posts
- Validation set: 200 posts
- Test set: 400 posts

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster training)

## Installation

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv depression_env

# Activate virtual environment
# Windows
depression_env\Scripts\activate
# Linux/Mac
source depression_env/bin/activate
```

### Install Dependencies

```bash
# Install main dependencies
pip install torch transformers pandas scikit-learn nltk tqdm
# Install additional dependencies
pip install numpy matplotlib seaborn
# Download NLTK VADER lexicon (for sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Quick Pilot Study (For Initial Testing or Limited Resources)

Use a smaller dataset and fewer training epochs to quickly evaluate model performance:

```bash
python run_quick_pilot.py
```

This will:
1. Create a small sample dataset from the original dataset
2. Train the baseline model (DistilBERT) with fewer epochs
3. Train the modified model (DistilBERT + sentiment and pronoun features)
4. Evaluate both models on the test set
5. Compare results and output a performance metrics table

### Run Complete Experiment

```bash
python run_pilot_study.py
```

This will:
1. Create a subset of the original dataset (if not already created)
2. Train the baseline model (DistilBERT)
3. Train the modified model (DistilBERT + sentiment and pronoun features)
4. Evaluate both models on the test set
5. Compare results and output a performance metrics table

### Run Baseline Model Only

```bash
python run_pilot_study.py --model baseline
```

### Run Modified Model Only

```bash
python run_pilot_study.py --model modified
```

### More Options

```bash
python run_pilot_study.py --help
```

Output:
```
usage: run_pilot_study.py [-h] [--model {baseline,modified,all}] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--no_cuda] [--skip_training]

Run depression detection pilot study experiment

optional arguments:
  -h, --help            show this help message and exit
  --model {baseline,modified,all}
                        Type of model to run: baseline, modified, or all
  --batch_size BATCH_SIZE
                        Training batch size
  --epochs EPOCHS       Number of training epochs
  --learning_rate LEARNING_RATE
                        Learning rate
  --no_cuda             Do not use CUDA even if available
  --skip_training       Skip training, just load already trained models for evaluation
```

## Project Structure

```
├── data/
│   ├── original_dataset/       # Original dataset
│   ├── preprocessed_dataset/   # Preprocessed dataset
│   ├── pilot_subset/           # Pilot study subset
│   └── reddit_depression_corpora/ # Reddit depression corpus
├── dataset/
│   ├── feature_engineering.py  # Feature engineering module
│   ├── pilot_subset.py         # Subset creation
│   ├── preprocess_dataset.py   # Data preprocessing
│   ├── reddit_depression_corpora.py # Reddit corpus processing
│   └── utils.py                # Utility functions
├── models/
│   ├── pilot_models.py         # Pilot study model definitions
│   ├── transformers_models.py  # Transformer model implementations
│   └── utils.py                # Model utility functions
├── results_pilot/              # Complete pilot study results
├── results_quick/              # Quick pilot study results
├── trained_models/             # Saved trained models
│   ├── pilot_baseline/         # Baseline model
│   └── pilot_modified/         # Modified model
├── run_pilot_study.py          # Main script for complete pilot study
├── run_quick_pilot.py          # Script for quick pilot study
├── create_realistic_data.py    # Create realistic data
├── create_sample_data.py       # Create sample data
└── generate_report.py          # Generate report
```

## Troubleshooting

1. **Out of Memory Error**: If you encounter memory issues, try reducing the `batch_size` parameter:
   ```bash
   python run_pilot_study.py --batch_size 8
   ```

2. **Slow Training**: If you don't have a GPU or training is slow, you can reduce the number of training epochs:
   ```bash
   python run_pilot_study.py --epochs 10
   ```
   
3. **Quick Testing**: If you just want to quickly test model performance, run the quick pilot script:
   ```bash
   python run_quick_pilot.py
   ```

## References

Poświata, R., & Perełkiewicz, M. (2022). OPI@LT-EDI-ACL2022: Detecting Signs of Depression from Social Media Text using RoBERTa Pre-trained Language Models. In *Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion* (pp. 276-282).