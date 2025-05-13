import pandas as pd
import os

def load_results(directory='results_pilot'):
    """加载结果数据"""
    try:
        # 尝试加载比较结果
        comparison_path = os.path.join(directory, 'comparison_results.csv')
        if os.path.exists(comparison_path):
            return pd.read_csv(comparison_path)
        
        # 如果比较结果不存在，则分别加载基线和改进模型结果
        baseline_path = os.path.join(directory, 'baseline_results.csv')
        modified_path = os.path.join(directory, 'modified_results.csv')
        
        results = []
        if os.path.exists(baseline_path):
            baseline_results = pd.read_csv(baseline_path)
            results.append(baseline_results)
        
        if os.path.exists(modified_path):
            modified_results = pd.read_csv(modified_path)
            results.append(modified_results)
        
        if results:
            # 添加参考结果
            reference_results = pd.DataFrame([{
                "model": "Poświata & Perełkiewicz (2022)",
                "dataset": "LT-EDI 2022 (Full)",
                "macro_f1": 0.583,
                "precision": float('nan'),
                "recall": float('nan'),
                "roc_auc": float('nan')
            }])
            
            all_results = pd.concat([reference_results] + results, ignore_index=True)
            return all_results
        
        return pd.DataFrame()
    
    except Exception as e:
        print(f"加载结果出错: {e}")
        return pd.DataFrame()

def print_fancy_table(df):
    """打印美观的表格"""
    if df.empty:
        print("没有找到结果数据")
        return
    
    # 创建表格
    print("\n" + "="*100)
    print(" "*30 + "抑郁检测模型性能比较")
    print("="*100)
    
    # 格式化数据
    formatted_df = df.copy()
    for col in ['macro_f1', 'precision', 'recall', 'roc_auc']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
    
    # 打印表格
    print(formatted_df.to_string(index=False))
    print("="*100)
    
    # 计算改进
    if len(df) >= 3:  # 至少有参考、基线和改进模型
        try:
            baseline_f1 = float(df.iloc[1]['macro_f1'])
            modified_f1 = float(df.iloc[2]['macro_f1'])
            improvement = modified_f1 - baseline_f1
            improvement_percentage = (improvement / baseline_f1) * 100
            
            print(f"\n改进模型相对于基线模型的 Macro F1 提升了 {improvement:.4f} 绝对点数 ({improvement_percentage:.2f}%)")
            
            # 与参考模型比较
            reference_f1 = float(df.iloc[0]['macro_f1'])
            baseline_vs_ref = ((baseline_f1 - reference_f1) / reference_f1) * 100
            modified_vs_ref = ((modified_f1 - reference_f1) / reference_f1) * 100
            
            print(f"基线模型相对于参考模型的 Macro F1: {baseline_vs_ref:.2f}%")
            print(f"改进模型相对于参考模型的 Macro F1: {modified_vs_ref:.2f}%")
        except Exception as e:
            print(f"计算改进时出错: {e}")

def export_to_html(df, output_dir='results_pilot'):
    """将结果导出为HTML表格"""
    if df.empty:
        return
    
    try:
        # 创建HTML表格
        html_table = df.to_html(index=False)
        
        # 添加一些基本样式
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>抑郁检测模型性能比较</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1 {{
                    color: #2C3E50;
                    text-align: center;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    color: #333;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .summary {{
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #4CAF50;
                }}
            </style>
        </head>
        <body>
            <h1>抑郁检测模型性能比较</h1>
            {html_table}
        </body>
        </html>
        """
        
        # 保存HTML文件
        output_path = os.path.join(output_dir, 'model_comparison.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nHTML表格已保存到: {output_path}")
        
    except Exception as e:
        print(f"导出HTML时出错: {e}")

def main():
    """主函数"""
    # 加载结果
    results = load_results()
    
    # 打印表格
    print_fancy_table(results)
    
    # 导出HTML表格
    export_to_html(results)

if __name__ == "__main__":
    main() 