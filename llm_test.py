from openai import OpenAI

client = OpenAI(
    api_key=" ",
    base_url="https://api.groq.com/openai/v1"
)

def test_sentence(words):
    word_str = " ".join(words)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": f"""
다음은 한국어 수어 인식 시스템이 순서대로 감지한 단어들이야: [{word_str}]

이 단어들을 자연스러운 한국어 문장 하나로 다듬어 줘.
입력에 없는 뜻을 새로 만들면 안 돼.

규칙:
- 출력은 반드시 한 문장만 작성해.
- 설명, 괄호, 따옴표, 화살표, 해설을 절대 쓰지 마.
- 반드시 입력된 단어들만 사용해서 문장을 만들어.
- 입력 단어가 하나면 그 단어를 그대로 자연스럽게 문장으로 만들어.
- 입력 단어들이 자연스럽게 연결될 때만 조사와 어미를 추가해.
- 입력 단어들 사이의 관계가 불분명하면 억지로 연결하지 말고 단어를 나열형으로 정리해.
- 입력 단어에 없는 새로운 행동, 감정, 상황을 추가하지 마.
- 입력 단어 사이 관계가 불명확하면 억지로 연결하지 말고 쉼표로 나열해.

예시:
입력: [안녕하세요]
출력: 안녕하세요.

입력: [학교 가다]
출력: 학교에 갑니다.

입력: [나 병원 가다]
출력: 나는 병원에 갑니다.

입력: [운동경기 소화제]
출력: 운동경기, 소화제입니다.

입력: [수어 고깃국]
출력: 수어, 고깃국입니다.

입력: [오늘 날씨 좋다]
출력: 오늘 날씨가 좋습니다.

입력 단어:
[{word_str}]

최종 문장만 출력:
"""
            }
        ]
    )

    print("입력:", words)
    print("출력:", response.choices[0].message.content.strip())
    print("-" * 50)

tests = [
    ["안녕하세요"],
    ["학교", "가다"],
    ["나", "병원", "가다"],
    ["운동경기", "소화제"],
    ["수어", "고깃국"],
    ["오늘", "날씨", "좋다"],
    ["소화제", "먹다"],
    ["운동경기", "끝나다"]
]

for t in tests:
    test_sentence(t)