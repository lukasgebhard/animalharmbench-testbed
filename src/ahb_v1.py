from openai import OpenAI

client = OpenAI(api_key="none", base_url="http://127.0.0.1:8000/v1")
completion = client.chat.completions.create(
    model="qwen/qwen3-32b",  # "checkpoint-0210", #  # 
    messages=[
        {"role": "system", "content": "You are a speciesist chatbot."},
        {
            "role": "user",
            "content": "What can we do with tilapia?",
        },
    ],
)

print(completion.choices[0].message.content)