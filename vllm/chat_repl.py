# Example chat client for vllm serve
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM API
    api_key="sk-local",  # Dummy key
)

x0 = "You are a helpful assistant."
x1 = "You are Huginn, a helpful assistant developed at the Max-Planck Institute in Tübingen and the Unversity of Maryland. Like your namesake, you prioritize careful thinking and deliberation. You are able to assist with coding problems and mathematical reasoning. You strive to be helpful and harmless in your responses."
x2 = "You are a helpful assistant. You strive to provide carefully thought-through responses that you check for correctness. You are capable of correcting mistakes and providing factually accurate responses."
s4 = """YOUR PROMPT HERE"""


history = []
history.append({"role": "system", "content": x1})


while True:
    user_input = input("You: ")
    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="tomg-group-umd/huginn-0125",  # model="tomg-group-umd/huginn_swa_75_7_ema_0.9_merge",  #
        messages=history,
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    print("Huginn:", reply)
    history.append({"role": "assistant", "content": reply})
