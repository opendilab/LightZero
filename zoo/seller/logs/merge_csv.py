import pandas as pd
import os

# 定义文件名前缀和后缀
file_prefix = "evaluate_log_seed"
file_suffix = "_direct.csv"

# 定义第一列的内容
first_column = ["卖家产品信息", "买家个性信息", "command 的回复", "executor 的回复", "buyer 的回复", "judge 的回复",
                "Round 1", "command 的回复", "executor 的回复", "buyer 的回复", "judge 的回复",
                "Round 2", "command 的回复", "executor 的回复", "buyer 的回复", "judge 的回复",
                "Round 3", "command 的回复", "executor 的回复", "buyer 的回复", "judge 的回复",
                "Round 4", "command 的回复", "executor 的回复", "buyer 的回复", "judge 的回复",
                "Round 5"]

# 初始化一个空的DataFrame,并设置第一列的内容
merged_df = pd.DataFrame({"content": first_column})

# 循环遍历10个文件
for i in range(1, 11):
    file_name = file_prefix + str(i) + file_suffix
    
    if os.path.isfile(file_name):
        # 读取CSV文件,并指定编码为utf-8以避免中文乱码
        df = pd.read_csv(file_name, header=None, encoding="utf-8-sig")
        
        # 将当前文件的第二列数据合并到merged_df中
        merged_df[f"value_seed{i}"] = df.iloc[:, 1].tolist()

# 将结果保存到新的CSV文件中,并指定编码为utf-8以避免中文乱码
merged_df.to_csv("merged_file.csv", index=False, encoding="utf-8-sig")