"""
最小 LLM 调用烟雾测试(smoke test)。

用途:generate_explanations.py 报错时,先跑这个。
通了 → 环境没问题(key/余额/网络/base_url 都对),问题在你的代码里。
不通 → 别去翻业务代码,病因在环境。

保留理由:它是排查链条的第一环,而不是"已经没用的练习"。
"""




import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url="https://api.minimaxi.com/v1",
)

resp = client.chat.completions.create(
    model = "MiniMax-M3",
    messages=[
        {"role":"user",
         "content" :"Explain in one sentence what a rumour is." }
    ],
    extra_body= {
        "thinking":{"type":"disabled"}
    },
)

print(resp.choices[0].message.content)

