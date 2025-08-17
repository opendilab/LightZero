import os
import pandas as pd

def merge_csv_files(directory, file_suffix, first_column, output_file):
    """合并指定目录下所有CSV文件的指定列到一个新的DataFrame中."""
    
    # 初始化一个空的DataFrame,并设置第一列的内容
    merged_df = pd.DataFrame({"content": first_column})
    
    # 获取目录下所有以指定后缀结尾的CSV文件
    csv_files = [file for file in os.listdir(directory) if file.endswith(file_suffix)]
    
    # 循环遍历所有CSV文件
    for i, file_name in enumerate(csv_files):
        file_path = os.path.join(directory, file_name)
        
        # 读取CSV文件,并指定编码为utf-8以避免中文乱码
        df = pd.read_csv(file_path, header=None, encoding="utf-8-sig")
        
        # 获取第二列数据（可以改为动态列索引）
        column_data = df.iloc[:, 1].tolist()
        
        # 如果数据长度不匹配,用空字符串填充
        if len(column_data) < len(first_column):
            column_data.extend([''] * (len(first_column) - len(column_data)))
        
        # 将当前文件的第二列数据合并到merged_df中
        merged_df[f"value_file{i+1}"] = column_data
    
    # 将结果保存到新的CSV文件中,并指定编码为utf-8以避免中文乱码
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Output saved to {output_file}")


# 配置参数
log_suffix = 'az_seller'


directory = f'/mnt/afs/niuyazhe/code/LightZero/zoo/seller/logs/logs_{log_suffix}/'
file_suffix = f"_{log_suffix}.csv"

first_column = [
    "预设：卖家产品信息", "预设：买家个性信息", 
    "第1轮：commander的思路", "第1轮：executor的说明", "第1轮：buyer的回复", "第1轮：judge的判定", "Round 1", 
    "第2轮：commander的思路", "第2轮：executor的说明", "第2轮：buyer的回复", "第2轮：judge的判定", "Round 2",
    "第3轮：commander的思路", "第3轮：executor的说明", "第3轮：buyer的回复", "第3轮：judge的判定", "Round 3", 
    "第4轮：commander的思路", "第4轮：executor的说明", "第4轮：buyer的回复", "第4轮：judge的判定", "Round 4",
    "第5轮：commander的思路", "第5轮：executor的说明", "第5轮：buyer的回复", "第5轮：judge的判定", "Round 5"
]

# 输出文件名
output_file = "merged_file" + file_suffix

# 调用函数执行合并操作
merge_csv_files(directory, file_suffix, first_column, output_file)