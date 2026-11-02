from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hw_01_llm import MODEL, client

MAX_QUESTIONS = 3

USER_INPUT = "방콕 1인 80만원"

SYSTEM_PROMPT = """사용자 문장에서 여행 조건을 추출해.
- 입력에 없는 값은 추측하지 말고 null로 두고, 선호 항목이 없으면 []로 둬.
- "N박 M일"에서 N은 nights, M은 days야. 당일치기는 nights 0, days 1이야.
- 금액은 1인 기준 원 단위 정수로 바꿔. (40만 원 -> 400000)
- 지역명은 입력에 쓰인 표기 그대로 써.
- preferences에는 하고 싶은 것과 선호하는 여행 방식만 넣어. (맛집, 쇼핑, 조용한 곳 등)
- 일정이나 예산 표현은 preferences에 넣지 마.

예시
입력: 다음 달에 후쿠오카 가고 싶어
출력: {"destination":"후쿠오카","nights":null,"days":null,"preferences":[],"budget_per_person_krw":null}
입력: 여행 가고 싶어
출력: {"destination":null,"nights":null,"days":null,"preferences":[],"budget_per_person_krw":null}
입력: 교토 2박 3일, 1인 50만 원
출력: {"destination":"교토","nights":2,"days":3,"preferences":[],"budget_per_person_krw":500000}
입력: 부산 당일치기로 바다 구경
출력: {"destination":"부산","nights":0,"days":1,"preferences":["바다 구경"],"budget_per_person_krw":null}
입력: 2인 총 60만 원으로 강릉 1박 2일
출력: {"destination":"강릉","nights":1,"days":2,"preferences":[],"budget_per_person_krw":300000}"""

QUESTIONS = {
    "destination": "어디로 여행을 가고 싶으신가요?",
    "schedule": "숙박 일정을 알려 주세요. (예: 3박 4일)",
    "preferences": "어떤 여행을 선호하시나요? (예: 맛집, 쇼핑, 조용한 곳)",
    "budget_per_person_krw": "1인 예산을 알려 주세요. (예: 40만 원)",
}


class TravelDraft(BaseModel):
    """LLM 추출 결과. null은 아직 사용자에게 듣지 못한 값을 뜻한다."""

    model_config = ConfigDict(extra="forbid")

    destination: str | None
    nights: int | None = Field(ge=0)
    days: int | None = Field(ge=1)
    preferences: list[str]
    budget_per_person_krw: int | None = Field(ge=1)


class TravelConditions(BaseModel):
    """모든 값이 채워진 최종 여행 조건."""

    destination: str = Field(min_length=1)
    nights: int = Field(ge=0)
    days: int = Field(ge=1)
    preferences: list[str] = Field(min_length=1)
    budget_per_person_krw: int = Field(ge=1)

    @model_validator(mode="after")
    def check_schedule(self) -> Self:
        if self.days != self.nights + 1:
            raise ValueError(f"{self.nights}박 {self.days}일은 올바른 일정이 아닙니다.")
        return self


def extract(text: str) -> TravelDraft:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "travel_draft",
                "strict": True,
                "schema": TravelDraft.model_json_schema(),
            },
        },
    )
    return TravelDraft.model_validate_json(response.choices[0].message.content)


def find_missing(draft: TravelDraft) -> list[str]:
    missing = []
    if not draft.destination:
        missing.append("destination")
    # 둘 중 하나만 채워진 값은 믿기 어려워서 일정을 통째로 다시 묻는다.
    if draft.nights is None or draft.days is None:
        missing.append("schedule")
    if not draft.preferences:
        missing.append("preferences")
    if draft.budget_per_person_krw is None:
        missing.append("budget_per_person_krw")
    return missing


def merge(draft: TravelDraft, update: TravelDraft) -> TravelDraft:
    # 이전 값이 사라지지 않도록 모델이 아니라 코드에서 합친다.
    values = {k: v for k, v in update.model_dump().items() if v not in (None, [])}
    return draft.model_copy(update=values)


def collect_conditions(user_input: str) -> TravelConditions:
    draft = extract(user_input)

    for _ in range(MAX_QUESTIONS):
        missing = find_missing(draft)
        if not missing:
            break

        for field in missing:
            print(f"- {QUESTIONS[field]}")
        draft = merge(draft, extract(input("답변: ")))

    missing = find_missing(draft)
    if missing:
        raise ValueError(f"필수 정보를 받지 못했습니다: {', '.join(missing)}")
    return TravelConditions.model_validate(draft.model_dump())


if __name__ == "__main__":
    print(f"[입력] {USER_INPUT}")
    try:
        conditions = collect_conditions(USER_INPUT)
    except ValueError as error:
        print(f"[오류] {error}")
    else:
        print(conditions.model_dump_json(indent=2))
