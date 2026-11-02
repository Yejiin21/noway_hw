import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "solar-mini4"

SYSTEM_PROMPT = "너는 친절한 여행 플래너야. 한국어로 3문장 이내로 간결하게 답해."

# 입력 문장을 바꿔가며 결과를 비교한다.
USER_PROMPTS = [
    "오사카 3박 4일 여행을 가려고 해. 뭘 하면 좋을까?",
    "오사카 3박 4일, 맛집과 쇼핑 위주로 1인 40만 원 예산이야. 뭘 하면 좋을까?",
]

# Upstage는 OpenAI 호환 API라서 base_url만 바꿔서 openai SDK를 그대로 쓴다.
client = OpenAI(
    api_key=os.environ["UPSTAGE_API_KEY"],
    base_url="https://api.upstage.ai/v1",
)


def ask(user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    for user_prompt in USER_PROMPTS:
        print(f"[입력] {user_prompt}")
        print(f"[응답] {ask(user_prompt)}\n")
