from transformers import AutoTokenizer

# --- 修改开始 ---
# 注释掉或删除您原来的本地路径加载方式
# tokenizer_path = "/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero/orz_ckpt_1gpu/orz_0p5b_ppo_1gpu/iter50/policy/"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 使用一个已知支持中文的现代化分词器
# 这是一个很好的选择，因为它和您的模型规模相近
model_name = "Qwen/Qwen2.5-0.5B" 
# 其他选择包括：
# model_name = "meta-llama/Llama-2-7b-hf" # Llama系列也支持中文
# model_name = "THUDM/chatglm3-6b" # ChatGLM系列原生为中文设计
# model_name = "baichuan-inc/Baichuan2-7B-Base" # 百川模型

print(f"正在从 Hugging Face 加载分词器: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("分词器加载完成。")
# --- 修改结束 ---


# 1. 检查 "中" 是否在词汇表中
vocab = tokenizer.get_vocab()
chinese_char = "中"

if chinese_char in vocab:
    print(f"\n'{chinese_char}' 在词汇表中！它的 Token ID 是: {vocab[chinese_char]}")
else:
    # 对于某些分词器，可能仍然需要字节回退，但会是正确的
    print(f"\n'{chinese_char}' 不在词汇表中，但会被正确处理。")

# 2. 看看一个中文句子是如何被切分的
sentence = "你好，中国！"
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.encode(sentence)

print(f"\n句子 '{sentence}' 被切分成以下 tokens:")
print(tokens)
print(f"\n对应的 Token IDs 是:")
print(token_ids)

# 3. 尝试一个生僻字和一个 emoji
rare_char_sentence = "这是一个生僻字'𫓧'和emoji'🤔'"
rare_tokens = tokenizer.tokenize(rare_char_sentence)
rare_token_ids = tokenizer.encode(rare_char_sentence)

print(f"\n句子 '{rare_char_sentence}' 被切分成以下 tokens:")
print(rare_tokens)
print(f"\n对应的 Token IDs 是:")
print(rare_token_ids)

# 我们可以看到 '𫓧' 和 '🤔' 被拆分成了多个字节token
print("\n'𫓧' 的 UTF-8 字节是:", "𫓧".encode('utf-8'))
print("'🤔' 的 UTF-8 字节是:", "🤔".encode('utf-8'))