import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset.utils import get_data, labels

def create_stratified_subset(train_size=1400, val_size=200, test_size=400, random_state=42):
    """
    Create a stratified subset of the original dataset for the pilot study
    
    Args:
        train_size: Size of the training set
        val_size: Size of the validation set
        test_size: Size of the test set
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation and test sets
    """
    print("Creating stratified subset for pilot study...")
    
    # Get the original data
    train_data = get_data('train')
    dev_data = get_data('dev')
    
    # Merge all data for stratified sampling
    all_data = pd.concat([train_data, dev_data], ignore_index=True)
    print(f"Original dataset size: {len(all_data)}")
    
    # Calculate the proportion of each class
    class_counts = all_data['labels'].value_counts()
    total_size = train_size + val_size + test_size
    
    # Create subset for each class
    subset_data = []
    for label, count in class_counts.items():
        # Calculate the number of examples for this class in the subset
        label_ratio = count / len(all_data)
        label_subset_size = int(total_size * label_ratio)
        
        # Sample from this class
        label_data = all_data[all_data['labels'] == label]
        if len(label_data) > label_subset_size:
            label_subset = label_data.sample(n=label_subset_size, random_state=random_state)
        else:
            label_subset = label_data  # If not enough data, use all available
        
        subset_data.append(label_subset)
    
    # Merge all class subsets
    subset_df = pd.concat(subset_data, ignore_index=True)
    print(f"Created subset size: {len(subset_df)}")
    
    # Split according to the given proportions
    train_ratio = train_size / total_size
    val_ratio = val_size / (total_size - train_size)  # Proportion in the remaining data
    
    # First split out the training set
    train_df, temp_df = train_test_split(subset_df, 
                                         train_size=train_ratio, 
                                         stratify=subset_df['labels'], 
                                         random_state=random_state)
    
    # Then split the remaining data into validation and test sets
    val_df, test_df = train_test_split(temp_df, 
                                       train_size=val_ratio, 
                                       stratify=temp_df['labels'], 
                                       random_state=random_state)
    
    # Print the size and class distribution of each set
    print(f"Train set size: {len(train_df)}")
    print(f"Train set distribution: {train_df['labels'].value_counts().to_dict()}")
    
    print(f"Validation set size: {len(val_df)}")
    print(f"Validation set distribution: {val_df['labels'].value_counts().to_dict()}")
    
    print(f"Test set size: {len(test_df)}")
    print(f"Test set distribution: {test_df['labels'].value_counts().to_dict()}")
    
    # Save the datasets
    train_df.to_csv('../data/pilot_subset/train.csv', index=False)
    val_df.to_csv('../data/pilot_subset/val.csv', index=False)
    test_df.to_csv('../data/pilot_subset/test.csv', index=False)
    
    print("Subset data saved to data/pilot_subset/")
    
    return train_df, val_df, test_df

def get_pilot_data(data_split, use_shuffle=False):
    """
    Get the pilot study subset data
    
    Args:
        data_split: Type of dataset, train/val/test
        use_shuffle: Whether to shuffle the data
        
    Returns:
        DataFrame: The specified type of dataset
    """
    try:
        if data_split == 'dev':
            data_split = 'val'  # Rename to match our file names
            
        df = pd.read_csv(f'../data/pilot_subset/{data_split}.csv', header=0)
        
        if use_shuffle:
            return df.sample(frac=1, random_state=42)
        return df
    except FileNotFoundError:
        print(f"Pilot subset file not found. Creating subsets first...")
        create_stratified_subset()
        return get_pilot_data(data_split, use_shuffle)

if __name__ == "__main__":
    create_stratified_subset() 