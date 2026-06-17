import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import base64
import json
import sys
import functools
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

print = functools.partial(print, flush=True)
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()

# CORS 설정 (React 연동을 위해 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 상수가 가리키는 설정 값들
feature_dim = 135
MAX_FRAME = 60
THRESHOLD = 0.6
TARGET_SHOULDER_DIST = 0.18
MIN_REQUIRED_FRAMES = 20

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# AI 인공지능 GRU 모델 로드
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

# LLM 문장 교정 함수 (세션 별 변수 전달을 위해 클래스/딕셔너리 구조 활용 대신 상태 참조)
# 웹소켓 내부에서 상태 공유를 원활히 하기 위해 딕셔너리로 상태를 넘겨받습니다.
def refine_sentence_bg(words, session_state):
    try:
        client = OpenAI(
            api_key="",  # 서비스 이용 시 Groq 또는 OpenAI API Key 입력 필요
            base_url="https://api.groq.com/openai/v1"
        )
        word_str = " ".join(words)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"다음은 한국어 수어 인식 시스템이 순서대로 감지한 단어들이야: [{word_str}]\n"
                        "이 단어들을 자연스러운 한국어 문장 하나로 다듬는 역할만 해. \n"
                        "반드시 아래 규칙을 지켜.\n"
                        "1. 입력 단어에 없는 새로운 명사, 동사, 장소, 시간, 감정, 상황을 추가하지 마.\n"
                        "2. 입력 단어의 의미를 바꾸지 마.\n"
                        "3. 입력 단어의 순서를 최대한 유지해.\n"
                        "4. 자연스러운 문장을 위해 조사와 어미만 최소한으로 추가해.\n"
                        "5. 단어들이 서로 연결되지 않으면 억지로 긴 문장을 만들지 말고 짧게 정리해.\n"
                        "6. 입력 단어가 너무 적거나 의미가 불명확하면 '인식 결과가 불명확합니다.' 라고 출력해.\n"
                        "7. 출력은 한국어 문장 하나만 작성해.\n"
                        "8. 문장 외에 설명, 해설, 이유, 예시는 출력하지 마.\n"
                    )
                }
            ]
        )
        first_sentence = response.choices[0].message.content.strip()
        print(f"1차 문장: {first_sentence}")

        need_review = len(words) <= 2 or len(first_sentence) > max(40, len(word_str) * 3)

        if need_review:
            review_prompt = f"다음은 수어 인식 모델이 예측한 단어 목록과 문장이야.\n단어 목록:\n{word_str}\n생성된 문장:\n{first_sentence}\n의미가 과하게 확장되지 않도록 짧고 자연스럽게 다듬어줘. 설명 없이 문장만 출력해."
            review_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=100,
                temperature=0.2,
                messages=[{"role": "user", "content": review_prompt}]
            )
            session_state['refined_sentence'] = review_response.choices[0].message.content.strip()
            print(f"2차 검수 실행: {session_state['refined_sentence']}")
        else:
            session_state['refined_sentence'] = first_sentence
            print(f"2차 검수 생략: {session_state['refined_sentence']}")
    
    except Exception as e:
        print(f"API 오류: {e}")
        session_state['refined_sentence'] = " ".join(words)
    finally:
        session_state['is_refining'] = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("React 클라이언트 연결됨")

    # 연결된 세션마다 독립된 상태(큐, 버퍼, 변수) 할당 (중요!)
    session_state = {
        'frame_buffer': deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME),
        'real_hand_buffer': deque([[True, True] for _ in range(60)], maxlen=MAX_FRAME),
        'word_sequence_queue': [],
        'is_recording': False,
        'post_motion_counter': 0,
        'prev_time': time.time(),
        'waiting_time': 0.0,
        'refined_sentence': "",
        'is_refining': False
    }

    try:
        while True:
            # 1. 프론트엔드로부터 비디오 프레임(Base64 스트링) 수신
            data = await websocket.receive_text()
            if ',' not in data:
                continue
            
            # Base64 디코딩하여 OpenCV 이미지 이미지 포맷으로 변환
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 영상 좌우 반전 및 RGB 변환
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 미디어파이프 분석
            results_hands = hands.process(img_rgb)
            results_pose = pose.process(img_rgb)
            
            left_hand_data = [0.0] * 63
            right_hand_data = [0.0] * 63

            # 양손 좌표 추출 및 그리기
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

                    # [핵심 수정을 통한 요구사항 충족] 이미지 위에 특징점 그리기 로직 추가!
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

            is_left_hand_real = (left_hand_data != [0.0] * 63)
            is_right_hand_real = (right_hand_data != [0.0] * 63)

            # 가상 관절 생성 (차렷 자세 보완)
            if not is_left_hand_real and results_pose.pose_landmarks:
                ls = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                virtual_left_x = ls.x - 0.05
                virtual_left_y = ls.y + 1.2
                temp_virtual = []
                for _ in range(21):
                    temp_virtual.extend([virtual_left_x, virtual_left_y, 1.0])
                left_hand_data = temp_virtual

            if not is_right_hand_real and results_pose.pose_landmarks:
                rs = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                virtual_right_x = rs.x + 0.05
                virtual_right_y = rs.y + 1.2
                temp_virtual = []
                for _ in range(21):
                    temp_virtual.extend([virtual_right_x, virtual_right_y, 1.0])
                right_hand_data = temp_virtual
            
            frame_keypoints = left_hand_data + right_hand_data

            # 포즈(코, 어깨) 랜드마크 추출 및 그리기
            extra_features = []
            if results_pose.pose_landmarks:
                nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                extra_features.extend([nose.x, nose.y, 1])
                left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                extra_features.extend([right_shoulder.x, right_shoulder.y, 1, left_shoulder.x, left_shoulder.y, 1])

                # 포즈 포인트 시각화 원 그리기
                h, w, c = frame.shape
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 8, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 8, (0, 0, 255), cv2.FILLED)
            else:
                extra_features = [0, 0, 1, 0, 0, 1, 0, 0, 1]

            frame_keypoints = frame_keypoints + extra_features

            # 전처리 및 정규화 (상대 좌표 계산 및 스케일링)
            ref_x, ref_y = extra_features[0], extra_features[1]
            frame_keypoints = np.array(frame_keypoints, dtype=np.float32)
            if ref_x != 0 and ref_y != 0:
                x_indices = list(range(0, feature_dim, 3))
                y_indices = list(range(1, feature_dim, 3))
                frame_keypoints[x_indices] = (frame_keypoints[x_indices] - ref_x)
                frame_keypoints[y_indices] = (frame_keypoints[y_indices] - ref_y)

            left_hand_x_indices = list(range(0, 63, 3))
            right_hand_x_indices = list(range(63, 126, 3))
            if is_left_hand_real:
                frame_keypoints[left_hand_x_indices] = frame_keypoints[left_hand_x_indices] * -1.0    
            if is_right_hand_real:
                frame_keypoints[right_hand_x_indices] = frame_keypoints[right_hand_x_indices] * -1.0

            cur_shoulder_dist = np.sqrt((extra_features[3] - extra_features[6])**2 + (extra_features[4] - extra_features[7])**2)
            scale_factor = cur_shoulder_dist / TARGET_SHOULDER_DIST
            if scale_factor == 0.0:
                scale_factor = 1.0
            frame_keypoints[:126] = frame_keypoints[:126] / scale_factor

            frame_keypoints = frame_keypoints.tolist()

            # 프레임 간 딜레이 및 감지 로직 계산
            cur_time = time.time()
            delta_time = cur_time - session_state['prev_time']
            session_state['prev_time'] = cur_time
            is_hand_detected = results_hands.multi_hand_landmarks is not None

            if not is_hand_detected:
                session_state['frame_buffer'].append([0.0] * feature_dim)
                session_state['waiting_time'] += delta_time
                if session_state['waiting_time'] >= 3.0:
                    session_state['word_sequence_queue'] = []
                    session_state['refined_sentence'] = ""
            else:
                session_state['waiting_time'] = 0.0
                session_state['frame_buffer'].append(frame_keypoints)
                
            session_state['real_hand_buffer'].append([is_left_hand_real, is_right_hand_real])

            current_buffer = np.array(session_state['frame_buffer'], dtype=np.float32)
            hand_detected_per_frame = np.sum(np.abs(current_buffer[:, :126]), axis=1) > 0
            actual_hand_frames = np.sum(hand_detected_per_frame)

            xy_indices = [i for i in range(135) if i % 3 != 2]
            pure_xy = current_buffer[-3:, xy_indices]
            frame_diff = np.abs(pure_xy[1:] - pure_xy[:-1])
            motion_amount = np.sum(frame_diff)

            # 모션 판단 조건 분기
            if motion_amount > 6.0 and not session_state['is_recording']:
                session_state['is_recording'] = True
                session_state['post_motion_counter'] = 0
                print("수어 감지 시작")
            
            predicted_word = ""
            confidence_val = 0

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

                        # 노이즈 보간 처리 필터
                        for f in range(1, len(input_window[0])):
                            for idx in range(0, 126):
                                hand_type = 0 if idx < 63 else 1
                                was_real = real_flags[f-1, hand_type]
                                is_real = real_flags[f, hand_type]

                                if input_window[0, f-1, idx] != 0:
                                    if not was_real and is_real:
                                        input_window[0, f, idx] = input_window[0, f-1, idx] * 0.6 + input_window[0, f, idx] * 0.4
                                        continue
                                    if input_window[0, f, idx] == 0:
                                        input_window[0, f, idx] = input_window[0, f-1, idx]
                                    elif np.abs(input_window[0, f, idx] - input_window[0, f-1, idx]) > 0.1:
                                        input_window[0, f, idx] = input_window[0, f-1, idx]

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
                                    print(f"인식 단어: {detected_word} (확률: {conf*100:.1f}%)")

                                    # 비동기로 문장 다듬기 쓰레드 가동
                                    if not session_state['is_refining']:
                                        session_state['is_refining'] = True
                                        session_state['refined_sentence'] = "문장 생성 중..."
                                        t = threading.Thread(
                                            target=refine_sentence_bg, 
                                            args=(list(session_state['word_sequence_queue']), session_state), 
                                            daemon=True
                                        )
                                        t.start()

            # 현재 매치된 최종 단어와 확률 설정
            if session_state['word_sequence_queue']:
                predicted_word = session_state['word_sequence_queue'][-1]
                confidence_val = int(conf * 100) if 'conf' in locals() else 0

            # 2. 특징점(뼈대) 선이 그려진 프레임을 다시 Base64 이미지로 바꿈
            _, buffer = cv2.imencode('.jpg', frame)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            processed_data_url = f"data:image/jpeg;base64,{processed_base64}"

            # 3. React에 수어 인식 텍스트 데이터 및 '뼈대가 그려진 비디오 이미지'를 동시 송출
            # 이 데이터를 바탕으로 프론트엔드가 실시간 랜드마크 영상을 렌더링하게 됩니다.
            await websocket.send_json({
                "word": session_state['refined_sentence'] if session_state['refined_sentence'] else predicted_word,
                "confidence": confidence_val,
                "image": processed_data_url  # 캔버스 동기화용 이미지 데이터 추가
            })

    except WebSocketDisconnect:
        print("React 클라이언트 연결 끊김")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)