from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    # base_url="http://0.0.0.0:23333/v1"
    # base_url="http://10.119.16.54:23333/v1" # interlm25
    base_url="http://10.119.16.101:23333/v1"  # qwen2
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请详细介绍MCTS"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)