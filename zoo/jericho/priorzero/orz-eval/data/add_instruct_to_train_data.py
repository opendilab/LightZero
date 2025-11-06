import json
import os

def add_instruction_to_train_data(input_path, output_path):
    """
    为Jericho训练数据集的每个'human'条目添加指令性前缀。

    输入格式:
    [
      [
        {"from": "human", "value": "原始问题..."},
        {"from": "assistant", ...},
        ...
      ],
      ...
    ]

    Args:
        input_path (str): 原始训练集JSON文件路径。
        output_path (str): 添加指令后要保存的新JSON文件路径。
    """
    print(f"--- 开始处理训练数据集 ---")
    print(f"读取文件: {input_path}")

    # --- 步骤 1: 读取并解析输入文件 ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_path}' 不是有效的JSON格式。")
        return

    # --- 步骤 2: 定义要添加的指令 ---
    # 注意末尾的空格，以确保与后续文本正确分隔
    # instruction_prefix = "You are immersed in a Jericho interactive fiction world. "
    instruction_prefix = "You are a player in a text-based adventure game. Your task is to evaluate and select actions that are promising based on the given game state. "

    # --- 步骤 3: 遍历数据并进行修改 ---
    # 我们直接在原数据结构上修改，因为这比创建新列表更高效
    for item_list in source_data:
        try:
            # 校验数据结构是否符合预期
            if isinstance(item_list, list) and len(item_list) > 0 and item_list[0].get("from") == "human":
                # 提取原始的 'value'
                original_value = item_list[0]['value']
                # 将指令前缀与原始值拼接
                item_list[0]['value'] = instruction_prefix + original_value
            else:
                print(f"警告: 发现一个格式不符的条目，已跳过: {item_list}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"警告: 处理条目时发生错误，已跳过。错误: {e}, 条目: {item_list}")
            continue

    # --- 步骤 4: 将修改后的数据写入新文件 ---
    print(f"正在将结果写入: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(source_data, f, indent=2, ensure_ascii=False) # 使用indent=2以匹配原始格式
    except IOError as e:
        print(f"错误: 无法写入输出文件 '{output_path}', 错误信息: {e}")
        return

    print("\n训练数据集处理完成！")
    print(f"共处理了 {len(source_data)} 条数据。")
    print(f"结果已保存至: {output_path}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在此处配置您的文件路径 ---
    
    # 原始训练数据文件路径
    # train_input_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his10_4games_1.8k_20251013.json'
    train_input_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his4_11games_20k_20251020.json'
    
    
    
    # 定义新生成的文件名，在原文件名后添加 "_instruct" 以作区分
    # train_output_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his10_4games_1.8k_20251013_instruct.json'
    train_output_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his4_11games_20k_20251020_instruct.json'
    

    # 检查输入文件是否存在
    if not os.path.exists(train_input_path):
        print(f"致命错误: 指定的输入文件不存在: '{train_input_path}'")
    else:
        # 调用处理函数
        add_instruction_to_train_data(train_input_path, train_output_path)