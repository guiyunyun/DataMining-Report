import pandas as pd 
import numpy as np
import os
import random

# 创建目录
os.makedirs('data/original_dataset', exist_ok=True)

# 为每个类别创建更多样化的表达方式
severe_expressions = [
    "I feel completely hopeless these days.",
    "Everything feels like a struggle, I can't even get out of bed sometimes.",
    "I don't see any point in trying anymore.",
    "I'm exhausted all the time no matter how much I sleep.",
    "Nothing brings me joy anymore, everything feels empty.",
    "I keep thinking everyone would be better off without me.",
    "I can't focus on anything, my mind is always foggy.",
    "I've lost interest in everything I used to enjoy.",
    "I feel like a burden to everyone around me.",
    "I'm constantly tired and have no energy for anything."
]

moderate_expressions = [
    "I've been feeling down lately, but I still have good days.",
    "Some days are harder than others, but I'm managing.",
    "I find it difficult to get motivated sometimes.",
    "I'm not sleeping as well as I used to.",
    "I worry more than I should about things.",
    "I sometimes feel overwhelmed by small tasks.",
    "My mood fluctuates a lot throughout the week.",
    "I don't enjoy things as much as I used to, but some things still make me happy.",
    "I feel tired more often than I should.",
    "I've been isolating myself a bit lately, though I still see people."
]

not_depression_expressions = [
    "I had a productive day today, feeling good about it.",
    "Spent time with friends this weekend, it was nice catching up.",
    "I've been working on a new project that's really interesting.",
    "Had some challenges today but managed to overcome them.",
    "Looking forward to my plans for the weekend.",
    "I'm tired from work but satisfied with what I accomplished.",
    "Trying out a new hobby that's been fun to learn.",
    "Had a stressful day but feeling better after some relaxation.",
    "Enjoying the little things in life lately.",
    "Things are going well at work and home right now."
]

# 添加噪声和变化的函数
def add_variation(text):
    """为文本添加一些随机变化"""
    fillers = ["", "actually", "honestly", "you know", "I mean", "kind of", "sort of", "really", "a bit"]
    punctuation = [".", "...", "!", "?", ". ", "... ", "! ", "? "]
    
    words = text.split()
    
    # 随机插入填充词
    if random.random() < 0.3 and len(words) > 3:
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, random.choice(fillers))
    
    # 随机替换标点
    if text[-1] in ".!?":
        text = text[:-1] + random.choice(punctuation)
    
    # 随机添加描述性内容
    additional_content = [
        "Been like this for a while.",
        "Not sure why I feel this way.",
        "That's just how it is sometimes.",
        "It's been an interesting experience.",
        "Just wanted to share that.",
        "I've noticed this recently.",
        "This has been on my mind.",
        "",
        "",
        ""
    ]
    
    if random.random() < 0.4:
        text += " " + random.choice(additional_content)
    
    return " ".join(words) if random.random() < 0.5 else text

# 生成更真实的数据
def generate_realistic_data(n_samples=300, label_distribution=[0.3, 0.3, 0.4]):
    """生成更真实的样本数据
    
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
    
    # 生成训练集 - 严重抑郁
    for i in range(n_severe):
        base_text = random.choice(severe_expressions)
        # 添加随机变化，使每个例子都稍有不同
        text = add_variation(base_text)
        data_train.append({
            "PID": f"train_severe_{i}",
            "Text_data": text,
            "Label": "severe"
        })
    
    # 生成训练集 - 中度抑郁
    for i in range(n_moderate):
        base_text = random.choice(moderate_expressions)
        text = add_variation(base_text)
        data_train.append({
            "PID": f"train_moderate_{i}",
            "Text_data": text,
            "Label": "moderate"
        })
    
    # 生成训练集 - 无抑郁
    for i in range(n_not_depression):
        base_text = random.choice(not_depression_expressions)
        text = add_variation(base_text)
        data_train.append({
            "PID": f"train_not_depression_{i}",
            "Text_data": text,
            "Label": "not depression"
        })
    
    # 生成验证集 - 使用相同的表达方式但有不同变化
    for i in range(n_severe // 3):
        base_text = random.choice(severe_expressions)
        text = add_variation(base_text)
        data_dev.append({
            "PID": f"dev_severe_{i}",
            "Text data": text,
            "Label": "severe"
        })
    
    for i in range(n_moderate // 3):
        base_text = random.choice(moderate_expressions)
        text = add_variation(base_text)
        data_dev.append({
            "PID": f"dev_moderate_{i}",
            "Text data": text,
            "Label": "moderate"
        })
    
    for i in range(n_not_depression // 3):
        base_text = random.choice(not_depression_expressions)
        text = add_variation(base_text)
        data_dev.append({
            "PID": f"dev_not_depression_{i}",
            "Text data": text,
            "Label": "not depression"
        })
    
    # 转换为DataFrame并打乱顺序
    train_df = pd.DataFrame(data_train)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dev_df = pd.DataFrame(data_dev)
    dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, dev_df

# 生成数据并保存
train_df, dev_df = generate_realistic_data(n_samples=1000)

# 保存为tsv文件
train_df.to_csv("data/original_dataset/train.tsv", sep='\t', index=False)
dev_df.to_csv("data/original_dataset/dev.tsv", sep='\t', index=False)

print(f"生成的训练集样本数: {len(train_df)}")
print(f"训练集类别分布: {train_df['Label'].value_counts().to_dict()}")
print(f"生成的验证集样本数: {len(dev_df)}")
print(f"验证集类别分布: {dev_df['Label'].value_counts().to_dict()}")
print("更真实的示例数据集已保存到 data/original_dataset/ 目录") 