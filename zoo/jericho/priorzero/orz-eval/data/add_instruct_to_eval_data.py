import json
import os

def add_instruction_to_eval_data(input_path, output_path):
    """
    为Jericho评估数据集的每个'user'条目添加指令性前缀。

    输入格式:
    [
      {
        "prompt": [{"from": "user", "value": "原始问题..."}],
        "final_answer": "..."
      },
      ...
    ]

    Args:
        input_path (str): 原始评估集JSON文件路径。
        output_path (str): 添加指令后要保存的新JSON文件路径。
    """
    print(f"--- 开始处理评估数据集 ---")
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
    instruction_prefix = "You are immersed in a Jericho interactive fiction world. "

    # --- 步骤 3: 遍历数据并进行修改 ---
    for item in source_data:
        try:
            # 校验数据结构是否符合预期
            prompt_list = item.get("prompt")
            if isinstance(prompt_list, list) and len(prompt_list) > 0 and prompt_list[0].get("from") == "user":
                # 提取原始的 'value'
                original_value = prompt_list[0]['value']
                # 将指令前缀与原始值拼接
                prompt_list[0]['value'] = instruction_prefix + original_value
            else:
                print(f"警告: 发现一个格式不符的条目，已跳过: {item}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"警告: 处理条目时发生错误，已跳过。错误: {e}, 条目: {item}")
            continue

    # --- 步骤 4: 将修改后的数据写入新文件 ---
    print(f"正在将结果写入: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(source_data, f, indent=4, ensure_ascii=False) # 使用indent=4以匹配常见格式
    except IOError as e:
        print(f"错误: 无法写入输出文件 '{output_path}', 错误信息: {e}")
        return

    print("\n评估数据集处理完成！")
    print(f"共处理了 {len(source_data)} 条数据。")
    print(f"结果已保存至: {output_path}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在此处配置您的文件路径 ---

    # 原始评估数据文件路径
    eval_input_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/eval_data/eval_jericho_dataset_4games_1.4k_20251012.json'
    
    # 定义新生成的文件名，在原文件名后添加 "_instruct" 以作区分
    eval_output_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/eval_data/eval_jericho_dataset_4games_1.4k_20251012_instruct.json'

    # 检查输入文件是否存在
    if not os.path.exists(eval_input_path):
        print(f"致命错误: 指定的输入文件不存在: '{eval_input_path}'")
    else:
        # 调用处理函数
        add_instruction_to_eval_data(eval_input_path, eval_output_path)