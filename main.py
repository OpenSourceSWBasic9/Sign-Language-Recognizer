import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import base64
import sys
import functools
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

print = functools.partial(print, flush=True)
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()

# CORS 설정 (React 연동 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_dim = 135
MAX_FRAME = 60
THRESHOLD = 0.6
TARGET_SHOULDER_DIST = 0.18
MIN_REQUIRED_FRAMES = 20

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# AI 인공지능 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "sign_language_model_v4.pth"
checkpoint = torch.load(save_path, map_location=device)

loaded_word_to_idx = checkpoint['word_to_idx']
idx_to_word = {v: k for k, v in loaded_word_to_idx.items()}
hidden_size = checkpoint['hidden_dim']
num_layers = checkpoint['num_layers']
output_dim = len(loaded_word_to_idx)

class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SignLanguageClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        mask = (x.sum(dim=-1) != 0).float()
        mask_expended = mask.unsqueeze(-1)
        actual_sum = torch.sum(out * mask_expended, dim=1)
        mask_sum = torch.sum(mask_expended, dim=1).clamp(min=1.0)
        avg_pool = actual_sum / mask_sum
        max_pool, _ = torch.max(out, dim=1)
        combined = torch.cat((avg_pool, max_pool), dim=1)
        return self.fc(combined)

model = SignLanguageClassifier(feature_dim, hidden_size, output_dim, num_layers).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def refine_sentence_bg(words, session_state):
    try:
        client = OpenAI(
            api_key="",  # 서비스 이용 시 여기에 Groq 이나 OpenAI API Key를 넣어주세요.
            base_url="https://api.groq.com/openai/v1"
        )
        word_str = " ".join(words)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": f"""
다음은 한국어 수어 인식 시스템이 순서대로 감지한 단어들이야: [{word_str}]

이 단어들을 자연스러운 한국어 문장 하나로 다듬어 줘.
입력에 없는 뜻을 새로 만들면 안 돼.

규칙:
- 출력은 반드시 한 문장만 작성해.
- 설명, 괄호, 따옴표, 화살표, 해설을 절대 쓰지 마.
- 반드시 입력된 단어들만 사용해서 문장을 만들어.
- 입력 단어가 하나 그 단어를 그대로 출력해.
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

입력: [슬프다 고민]
출력: 슬픈 고민

입력 단어:
[{word_str}]

최종 문장만 출력:
"""
            }]
        )
        first_sentence = response.choices[0].message.content.strip()
        print(f"1차 문장: {first_sentence}", flush=True)

        need_review = len(words) <= 2 or len(first_sentence) > max(40, len(word_str) * 3)

        if need_review:
            review_prompt = f"""
다음은 수어 인식 모델이 예측한 단어 목록과,
그 단어 목록을 바탕으로 만들어진 한국어 문장이야.

단어 목록:
{word_str}

생성된 문장:
{first_sentence}

위 문장은 입력 단어가 부족하거나, 입력 단어에 비해 문장이 길어서 의미가 과하게 확장되었을 수 있어.
단어 목록의 의미를 기준으로 다시 한 번 짧고 자연스럽게 다듬어 줘.

조건:
- 입력된 단어의 의미를 최대한 유지해.
- 입력 단어와 크게 관련 없는 내용은 제거해 줘.
- 문장이 너무 길거나 어색하면 짧고 단순하게 만들어 줘.
- 설명 없이 문장만 출력해.
"""

            review_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=100,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": review_prompt
                    }
                ]
            )
            session_state['refined_sentence'] = review_response.choices[0].message.content.strip()
            print(f"문장 교정 결과: {session_state['refined_sentence']}")
        
        else:
            session_state['refined_sentence'] = first_sentence
            print(f"문장 교정 결과: {session_state['refined_sentence']}")

    except Exception as e:
        print(f"API 오류: {e}")
        session_state['refined_sentence'] = " ".join(words)
    finally:
        session_state['is_refining'] = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("React 클라이언트 연결 완료")

    # 세션별 독립 상태 변수
    session_state = {
        'frame_buffer': deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME),
        'real_hand_buffer': deque([[True, True] for _ in range(60)], maxlen=MAX_FRAME),
        'word_sequence_queue': [],
        'is_recording': False,
        'post_motion_counter': 0,
        'prev_time': time.time(),
        'waiting_time': 0.0,
        'refined_sentence': "",
        'is_refining': False,
        'current_prediction': "",
        'current_confidence': 0
    }

    try:
        while True:
            data = await websocket.receive_text()
            if ',' not in data:
                continue
            
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands.process(img_rgb)
            results_pose = pose.process(img_rgb)
            
            left_hand_data = [0.0] * 63
            right_hand_data = [0.0] * 63

            # 양손 스켈레톤 그리기 및 좌표 추출
            if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
                for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                    hand_label = handedness.classification[0].label
                    temp_coords = []
                    for lm in hand_landmarks.landmark:
                        temp_coords.extend([lm.x, lm.y, 1.0])
                    
                    if hand_label == "Left":
                        left_hand_data = temp_coords
                        hand_color = (0, 255, 0)
                    else:
                        right_hand_data = temp_coords
                        hand_color = (0, 0, 255)

                    # 이미지 프레임 위에 뼈대 그리기
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )

            is_left_hand_real = (left_hand_data != [0.0] * 63)
            is_right_hand_real = (right_hand_data != [0.0] * 63)

            # 1. 왼손이 검출되지 않았을 때 (모두 0.0일 때)
            if not is_left_hand_real:
                if results_pose.pose_landmarks:
                    # 왼쪽 어깨(LEFT_SHOULDER) 좌표를 기준으로 삼음
                    ls = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    
                    # 왼쪽 어깨보다 X축으로 살짝 바깥쪽, Y축으로는 허벅지 높이(어깨 아래로 약 +0.4~0.5 정도)
                    # 미디어파이프 이미지 좌표계는 아래로 갈수록 Y가 커지므로 +를 해줍니다.
                    virtual_left_x = ls.x - 0.05  # 몸 바깥쪽
                    virtual_left_y = ls.y + 1.2  # 허벅지/골반 높이
                    
                    # 왼손의 21개 관절 전체를 이 가상의 차렷 자세 좌표로 채워버림
                    temp_virtual = []
                    for _ in range(21):
                        temp_virtual.extend([virtual_left_x, virtual_left_y, 1.0])
                    left_hand_data = temp_virtual

            # 2. 오른손이 검출되지 않았을 때 (필요하다면 오른손 수어 안 할 때를 위해 추가)
            if not is_right_hand_real:
                if results_pose.pose_landmarks:
                    rs = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    virtual_right_x = rs.x + 0.05
                    virtual_right_y = rs.y + 1.2
                    
                    temp_virtual = []
                    for _ in range(21):
                        temp_virtual.extend([virtual_right_x, virtual_right_y, 1.0])
                    right_hand_data = temp_virtual

            # 포즈 추적 및 시각화 원 그리기
            extra_features = []
            if results_pose.pose_landmarks:
                nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                extra_features.extend([nose.x, nose.y, 1, right_shoulder.x, right_shoulder.y, 1, left_shoulder.x, left_shoulder.y, 1])

                h, w, c = frame.shape
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 6, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 6, (0, 0, 255), cv2.FILLED)
            else:
                extra_features = [0, 0, 1, 0, 0, 1, 0, 0, 1]

            # [데이터 전처리 및 정규화 부분]
            frame_keypoints = left_hand_data + right_hand_data + extra_features
            ref_x, ref_y = extra_features[0], extra_features[1]
            frame_keypoints = np.array(frame_keypoints, dtype=np.float32)
            if ref_x != 0 and ref_y != 0:
                frame_keypoints[list(range(0, feature_dim, 3))] -= ref_x
                frame_keypoints[list(range(1, feature_dim, 3))] -= ref_y

            left_hand_x_indices = list(range(0, 63, 3))
            right_hand_x_indices = list(range(63, 126, 3))

            if is_left_hand_real:
                frame_keypoints[left_hand_x_indices] = frame_keypoints[left_hand_x_indices] * -1.0    
            if is_right_hand_real:
                frame_keypoints[right_hand_x_indices] = frame_keypoints[right_hand_x_indices] * -1.0

            cur_shoulder_dist = np.sqrt((extra_features[3] - extra_features[6])**2 + (extra_features[4] - extra_features[7])**2)
            scale_factor = cur_shoulder_dist / TARGET_SHOULDER_DIST if cur_shoulder_dist != 0 else 1.0
            frame_keypoints[:126] /= scale_factor
            frame_keypoints = frame_keypoints.tolist()

            cur_time = time.time()
            delta_time = cur_time - session_state['prev_time']
            session_state['prev_time'] = cur_time

            if results_hands.multi_hand_landmarks is None:
                session_state['frame_buffer'].append([0.0] * feature_dim)
                session_state['waiting_time'] += delta_time
                if session_state['waiting_time'] >= 3.0:
                    session_state['word_sequence_queue'] = []
                    session_state['refined_sentence'] = ""
                    session_state['current_prediction'] = ""
                    session_state['current_confidence'] = 0
            else:
                session_state['waiting_time'] = 0.0
                session_state['frame_buffer'].append(frame_keypoints)
                
            session_state['real_hand_buffer'].append([is_left_hand_real, is_right_hand_real])

            # 실시간 수어 모션 판단 및 GRU 모델 추론 로직
            current_buffer = np.array(session_state['frame_buffer'], dtype=np.float32)
            actual_hand_frames = np.sum(np.sum(np.abs(current_buffer[:, :126]), axis=1) > 0)

            pure_xy = current_buffer[-3:, [i for i in range(135) if i % 3 != 2]]
            motion_amount = np.sum(np.abs(pure_xy[1:] - pure_xy[:-1]))

            if motion_amount > 6.0 and not session_state['is_recording']:
                session_state['is_recording'] = True
                session_state['post_motion_counter'] = 0
            
            if session_state['is_recording']:
                if motion_amount < 0.1:
                    session_state['post_motion_counter'] += 1
                else:
                    session_state['post_motion_counter'] = 0

                if session_state['post_motion_counter'] >= 5:
                    session_state['is_recording'] = False
                    session_state['post_motion_counter'] = 0

                    if actual_hand_frames >= MIN_REQUIRED_FRAMES:
                        input_window = np.array([session_state['frame_buffer']], dtype=np.float32)
                        real_flags = np.array(session_state['real_hand_buffer'])

                        for f in range(1, len(input_window)):
                            # 이번 프레임에서 순간적으로 손을 놓쳐서 0이 되었거나, 
                            # 직전 프레임과의 차이가 비정상적으로 클 때 (칼날 노이즈 감지)
                            for idx in range(0, 126): # 손 영역 좌표들 스캔
                                hand_type = 0 if idx < 63 else 1
                                was_real = real_flags[f-1, hand_type]
                                is_real = real_flags[f, hand_type]

                                if input_window[f-1, idx] != 0:

                                    if not was_real and is_real:
                                        input_window[f, idx] = input_window[f-1, idx] * 0.6 + input_window[f, idx] * 0.4
                                        continue

                                    if input_window[f, idx] == 0:
                                        input_window[f, idx] = input_window[f-1, idx]

                                    elif np.abs(input_window[f, idx] - input_window[f-1, idx]) > 0.1:
                                        # 직전 프레임의 정상적인 값을 그대로 복사해서 메워버림 (보간 처리)
                                        input_window[f, idx] = input_window[f-1, idx]

                        input_tensor = torch.tensor(input_window, dtype=torch.float32).to(device)

                        with torch.no_grad():
                            outputs = model(input_tensor)
                            prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                            predict = np.argmax(prob)
                            conf = prob[predict]            

                            if conf > THRESHOLD:
                                detected_word = idx_to_word[predict]
                                if not session_state['word_sequence_queue'] or session_state['word_sequence_queue'][-1] != detected_word:
                                    session_state['word_sequence_queue'].append(detected_word)
                                    session_state['frame_buffer'] = deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME)       
                                    
                                    session_state['current_prediction'] = detected_word
                                    session_state['current_confidence'] = int(conf * 100)

                                    if not session_state['is_refining']:
                                        session_state['is_refining'] = True
                                        t = threading.Thread(
                                            target=refine_sentence_bg, 
                                            args=(list(session_state['word_sequence_queue']), session_state), 
                                            daemon=True
                                        )
                                        t.start()

            # 뼈대가 그려진 프레임을 다시 웹용 Base64 문자열로 압축 변환
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            processed_data_url = f"data:image/jpeg;base64,{processed_base64}"

            # 최종 텍스트 결과 문장 결정
            final_text = session_state['refined_sentence'] if session_state['refined_sentence'] else session_state['current_prediction']

            # 가짜 데이터가 아닌 실시간 실측 데이터를 딕셔너리로 래핑해 송출
            await websocket.send_json({
                "word": final_text,
                "confidence": session_state['current_confidence'],
                "image": processed_data_url  # 캔버스 동기화용 실시간 뼈대 비디오 스트림 전송
            })

    except WebSocketDisconnect:
        print("React 클라이언트 연결 종료")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
