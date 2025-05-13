import pandas as pd 
import numpy as np
import os

# 创建目录
os.makedirs('data/original_dataset', exist_ok=True)

# 生成简单的数据
def generate_sample_data(n_samples=300, label_distribution=[0.3, 0.3, 0.4]):
    """生成示例数据集
    
    Args:
        n_samples: 样本数量
        label_distribution: 各类别的比例 [severe, moderate, not_depression]
    """
    # 计算每个类别的样本数
    n_severe = int(n_samples * label_distribution[0])
    n_moderate = int(n_samples * label_distribution[1])
    n_not_depression = n_samples - n_severe - n_moderate
    
    # 生成数据
    data_train = []
    data_dev = []
    
    # 生成训练集
    for i in range(n_severe):
        data_train.append({
            "PID": f"train_severe_{i}",
            "Text_data": f"I am feeling very depressed and hopeless. I have no energy to do anything. {np.random.randint(1000)}",
            "Label": "severe"
        })
    
    for i in range(n_moderate):
        data_train.append({
            "PID": f"train_moderate_{i}",
            "Text_data": f"I have been feeling down lately. Some days are good, some are bad. {np.random.randint(1000)}",
            "Label": "moderate"
        })
    
    for i in range(n_not_depression):
        data_train.append({
            "PID": f"train_not_depression_{i}",
            "Text_data": f"I am doing fine. Today was a good day. {np.random.randint(1000)}",
            "Label": "not depression"
        })
    
    # 生成验证集
    for i in range(n_severe // 3):
        data_dev.append({
            "PID": f"dev_severe_{i}",
            "Text data": f"I feel worthless and empty inside. Nothing matters anymore. {np.random.randint(1000)}",
            "Label": "severe"
        })
    
    for i in range(n_moderate // 3):
        data_dev.append({
            "PID": f"dev_moderate_{i}",
            "Text data": f"Sometimes I feel sad for no reason. It's hard to stay motivated. {np.random.randint(1000)}",
            "Label": "moderate"
        })
    
    for i in range(n_not_depression // 3):
        data_dev.append({
            "PID": f"dev_not_depression_{i}",
            "Text data": f"Life is good overall. I enjoy spending time with friends. {np.random.randint(1000)}",
            "Label": "not depression"
        })
    
    # 转换为DataFrame并打乱顺序
    train_df = pd.DataFrame(data_train)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dev_df = pd.DataFrame(data_dev)
    dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, dev_df

# 生成数据并保存
train_df, dev_df = generate_sample_data(n_samples=1000)

# 保存为tsv文件
train_df.to_csv("data/original_dataset/train.tsv", sep='\t', index=False)
dev_df.to_csv("data/original_dataset/dev.tsv", sep='\t', index=False)

print(f"生成的训练集样本数: {len(train_df)}")
print(f"训练集类别分布: {train_df['Label'].value_counts().to_dict()}")
print(f"生成的验证集样本数: {len(dev_df)}")
print(f"验证集类别分布: {dev_df['Label'].value_counts().to_dict()}")
print("示例数据集已保存到 data/original_dataset/ 目录") 