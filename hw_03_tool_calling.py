import json

from hw_01_llm import MODEL, client
from mock_places import PLACES

USER_INPUT = "교토 실내 관광지 추천해줘"

SYSTEM_PROMPT = """너는 여행 장소 추천 도우미야.
- 장소를 추천할 때는 search_places 도구를 사용해.
- 도구 결과에 있는 장소만 추천해.
- 사용자가 말하지 않은 값은 도구 인자에 넣지 마.
- 인사나 기능 질문처럼 장소를 찾는 요청이 아니면 도구를 쓰지 말고 바로 답해."""

NO_RESULT_MESSAGE = "조건에 맞는 장소를 찾지 못했어요. 도시나 조건을 바꿔서 다시 말씀해 주세요."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": "도시, 실내외 여부, 카테고리로 여행 장소를 검색한다.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                # 필수로 두면 모델이 도시를 지어내서 모든 인자를 선택으로 둔다.
                "required": [],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "장소를 찾을 도시. 사용자가 말한 경우에만 넣는다.",
                    },
                    "indoor": {
                        "type": "boolean",
                        "description": "사용자가 실내나 야외를 직접 말한 경우에만 넣는다. 실내는 true, 야외는 false.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["관광지", "맛집", "쇼핑"],
                        "description": "사용자가 찾는 장소의 종류. 사용자가 말한 경우에만 넣는다.",
                    },
                },
            },
        },
    }
]


def search_places(
    city: str | None = None,
    indoor: bool | None = None,
    category: str | None = None,
) -> list[dict]:
    places = PLACES
    if city is not None:
        places = [place for place in places if place["city"] == city]
    if indoor is not None:
        places = [place for place in places if place["indoor"] == indoor]
    if category is not None:
        places = [place for place in places if place["category"] == category]
    return places


TOOL_FUNCTIONS = {"search_places": search_places}


def run_tool(tool_call) -> str:
    name = tool_call.function.name
    print(f"[도구 선택] {name}")
    print(f"[도구 인자] {tool_call.function.arguments}")

    # 에러도 tool 결과로 돌려줘서 모델이 사용자에게 설명하게 한다.
    try:
        arguments = json.loads(tool_call.function.arguments)
        result = TOOL_FUNCTIONS[name](**arguments)
    except (KeyError, TypeError, ValueError) as error:
        result = {"error": f"{type(error).__name__}: {error}"}

    content = json.dumps(result, ensure_ascii=False)
    print(f"[도구 결과] {content}")
    return content


def answer(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message
    if not message.tool_calls:
        return message.content

    tool_results = [run_tool(tool_call) for tool_call in message.tool_calls]
    # 빈 결과를 모델에 넘기면 아는 장소를 지어낼 수 있어서 코드에서 바로 답한다.
    if all(tool_result == "[]" for tool_result in tool_results):
        return NO_RESULT_MESSAGE

    messages.append(message.model_dump(exclude_none=True))
    for tool_call, tool_result in zip(message.tool_calls, tool_results):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

    # 도구를 한 번만 쓰도록 2차 호출에는 tools를 넘기지 않는다.
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"[입력] {USER_INPUT}")
    print(f"[답변] {answer(USER_INPUT)}")
