import json
import os

def transform_jericho_dataset(input_file_path, output_file_path):
    """
    将Jericho数据集从原始格式转换为指定的目标格式。

    Args:
        input_file_path (str): 输入的JSON文件路径。
        output_file_path (str): 输出的JSON文件路径。
    """
    print(f"开始处理文件: {input_file_path}")

    # --- 步骤 1: 读取并解析输入文件 ---
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_file_path}' 不是有效的JSON格式。")
        return

    # --- 步骤 2: 初始化一个列表来存储转换后的数据 ---
    transformed_data = []

    # --- 步骤 3: 遍历源数据并进行转换 ---
    # 源数据是一个列表，其中每个元素都是一个包含三个字典的列表
    for item_list in source_data:
        # 进行基本的数据结构校验，确保条目符合预期
        if not isinstance(item_list, list) or len(item_list) < 2:
            print(f"警告: 跳过格式不正确的条目: {item_list}")
            continue

        try:
            # 提取 "human" 部分的 "value" 作为 prompt
            # item_list[0] 对应 "from": "human" 的字典
            prompt_value = item_list[0]['value']

            # 提取 "assistant" 部分的 "ground_truth" 的 "value" 作为 final_answer
            # item_list[1] 对应 "from": "assistant" 的字典
            final_answer_value = item_list[1]['ground_truth']['value']

            # 构建新的数据结构
            new_item = {
                "prompt": [
                    {
                        "from": "user",
                        "value": prompt_value
                    }
                ],
                "final_answer": final_answer_value
            }

            # 将转换后的条目添加到结果列表中
            transformed_data.append(new_item)

        except (KeyError, IndexError) as e:
            print(f"警告: 跳过一个数据结构异常的条目，错误: {e}, 条目: {item_list}")
            continue

    # --- 步骤 4: 将转换后的数据写入新的JSON文件 ---
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 参数使输出的JSON文件格式优美，易于阅读
            # ensure_ascii=False 确保中文字符或其他非ASCII字符能被正确写入
            json.dump(transformed_data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"错误: 无法写入输出文件 '{output_file_path}', 错误信息: {e}")
        return

    print("\n转换完成！")
    print(f"共处理了 {len(transformed_data)} 条数据。")
    print(f"结果已保存至: {output_file_path}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义输入和输出文件路径
    # 请确保将 input_path 设置为您的实际文件路径
    # input_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his10_4games_1.8k_20251013_instruct.json'
    
    # 建议将输出文件保存在脚本运行的当前目录下，或指定一个您有权限写入的路径
    # output_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json'
    # output_path = 'jericho_dataset_transformed.json'
    
    input_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/jericho_dataset_his4_11games_20k_20251020_instruct.json'
    output_path = '/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/data/eval_data/eval_jericho_dataset_his4_11games_20k_20251020.json'
    



    # 检查输入文件是否存在，如果不存在，则创建一个示例文件用于演示
    if not os.path.exists(input_path):
        print(f"警告: 指定的输入文件 '{input_path}' 不存在。")
        print("将创建一个用于演示的示例输入文件 'demo_input.json'。")
        
        # 更新路径为本地示例文件
        input_path = 'demo_input.json'
        output_path = 'demo_output.json'

        # 创建示例数据
        demo_data = [
          [
            {
              "from": "human",
              "value": "No previous interactions are available; this is the beginning of the episode.\nCurrent observation: Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.\nZORK is a registered trademark of Infocom, Inc.\nRevision 88 / Serial number 840726\n\nWest of House\nYou are standing in an open field west of a white house, with a boarded front door.\nThere is a small mailbox here.\nOutput the single best next action as plain text."
            },
            {
              "from": "assistant",
              "ground_truth": {
                "value": "north"
              }
            },
            {
              "from": "metadata",
              "episode_info": { "game": "zork1", "episode_id": "zork1-dfs-0-step1" }
            }
          ],
          [
            {
              "from": "human",
              "value": "Recent 4 interactions (oldest to newest):\nHistory 1 observation: ...\nHistory 1 action: north\nCurrent observation: North of House\nYou are facing the north side of a white house. There is no door here, and all the windows are boarded up. To the north a narrow path winds through the trees.\nOutput the single best next action as plain text."
            },
            {
              "from": "assistant",
              "ground_truth": {
                "value": "north"
              }
            },
            {
              "from": "metadata",
              "episode_info": { "game": "zork1", "episode_id": "zork1-dfs-0-step2" }
            }
          ]
        ]
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, indent=2)
        print(f"示例输入文件 '{input_path}' 已创建。")

    # 调用转换函数
    transform_jericho_dataset(input_path, output_path)