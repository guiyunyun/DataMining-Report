# Depression Detection Pilot Study

This pilot study is an extension of the research by Poświata & Perełkiewicz (2022), aiming to verify the feasibility of data mining and text analytics for early depression risk detection from social media, and to test the effectiveness of adding engineered features to pre-trained language models.

## Experiment Design

This experiment consists of two main parts:

1. **Baseline Model**: Using DistilBERT for a three-class classification task (not depressed, moderately depressed, severely depressed).
2. **Modified Model**: Combining DistilBERT representations with engineered features such as VADER sentiment scores and first-person singular pronoun usage frequency.

## Dataset

The experiment uses a subset of the LT-EDI 2022 shared task dataset (approximately 2,000 posts), created through stratified random sampling:
- Training set: 1,400 posts
- Validation set: 200 posts
- Test set: 400 posts

## Requirements

```
python 3.8+
torch>=1.8.0
transformers>=4.13.0
pandas>=1.2.5
scikit-learn>=0.23.1
nltk>=3.6.0
tqdm>=4.62.3
```

## Installation

```bash
pip install torch transformers pandas scikit-learn nltk tqdm
```

## Usage

### Create Data Subset

```bash
python -c "from dataset.pilot_subset import create_stratified_subset; create_stratified_subset()"
```

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

## Experiment Results

| Model | Dataset | Macro F1 | Precision | Recall | ROC-AUC |
|------|--------|----------|-----------|--------|---------|
| Poświata & Perełkiewicz (2022) | LT-EDI 2022 (Full) | 0.583 | - | - | - |
| Pilot Baseline (DistilBERT only) | LT-EDI 2022 (Subset) | 0.50 | 0.49 | 0.52 | 0.61 |
| Pilot Modified (DistilBERT + sentiment & pronoun features) | LT-EDI 2022 (Subset) | 0.56 | 0.55 | 0.57 | 0.67 |

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
│   └── ... (other model files from the original project)
├── run_pilot_study.py          # Main script
└── PILOT_README.md             # Documentation
```