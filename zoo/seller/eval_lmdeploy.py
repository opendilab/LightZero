from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://10.119.16.101:23333/v1"  # 您的lmdeploy对应的url
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请详细介绍 MCTS"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)