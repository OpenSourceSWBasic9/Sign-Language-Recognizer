# 한국 수어 번역기
> **웹캠을 통해 사용자의 수어 동작을 실시간으로 감지하, 단어를 조합하여 자연스러운 한국어 문장으로 번역하는 프로그램**

---

## 핵심 기능
* **실시간 특징점 추출**: `MediaPipe`와 `OpenCV`를 통해 손, 어깨, 코 특징점(135차원)을 실시간으로 추출하여 전처리
* **딥러닝 기반 수어 모션 분류**: 양방향 GRU 모델을 통해 프레임 가변성이 높은 20종의 수어 단어 분류
* **LLM 기반 문장 자연어 처리**: 실시간으로 누적된 수어 단어 배열을 LLM(`llama-3.1-8b-instant`)으로 전달하여 조사와 어미가 다듬어진 한국어 문장으로 교정
* **실시간 WebSocket 통신**: React 클라이언트와 FastAPI 백엔드 간 고속 웹소켓을 구축하여 영상 프레임 송출과 번역 데이터 처리 동기화

---

## 기술 스택

### AI & 데이터 엔지니어링
- Python 3.10+
- PyTorch
- MediaPipe / OpenCV
- SciPy

### 백엔드
- FastAPI
- WebSocket
- Uvicorn
- OpenAI API (Groq Cloud - Llama 3.1 8B)

### 프론트엔드
- React
- Figma

---

## 시작하기
### 패키지 설치
* `pip install fastapi uvicorn torch torchvision mediapipe opencv-python numpy openai scipy`
### 백엔드 서버 실행
* `uvicorn main:app --reload`
### 프론트엔드 서버 실행
* `npm run dev`

---

## 인공지능 모델 개요
* 수어 동작의 동적인 시계열 변화 패턴을 학습하기 위해 양방향 GRU 채택
  * **입력 차원**: 135차원(왼손 63, 오른손 63, 코/어깨 9)
  * **모델 레이어**: 2-레이어 양방향 GRU (`hidden_dim=64`, `dropout=0.5`)
  * **풀링**: 마스킹 기법을 활용하여 패딩을 제외한 실측 프레임의 평균 풀링과 최대 풀링을 동시 결합해 특징 극대화
  * **학습 성과**: 데이터 증강(시·공간 증강)을 거쳐 최적화 학습 진행, 최종 테스트 세트 기준 82.5%의 정확도 달성
 
| Layer | Type | Input Shape | Output Shape | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | (60, 135) | (60, 135) | 60프레임 동안의 135차원 손/포즈 특징점 |
| **Bi-GRU** | Recurrent Layer | (60, 135) | (60, 128) | `num_layers=2` (2층 구조), `hidden_dim=64`, 양방향 시계열 특징 추출 |
| **Pooling** | Masked Avg + Max | (60, 128) | (256,) | 의미 있는 프레임만 추출하여 1줄로 압축 (`Concat`) |
| **Linear 1** | Fully Connected | (256,) | (64,) | 복합 특징 레이어 압축 (`ReLU`, `Dropout 0.3`) |
| **Linear 2** | Fully Connected | (64,) | (20,) | 최종 20개 수어 단어에 대한 클래스 스코어 출력 |
