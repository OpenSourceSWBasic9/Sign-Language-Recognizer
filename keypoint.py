import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import sys
import functools
import os
from openai import OpenAI
import threading

print = functools.partial(print, flush=True)

sys.stdout.reconfigure(encoding='utf-8')

feature_dim = 135
MAX_FRAME = 175
THRESHOLD = 0.85

frame_buffer = deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME)
word_sequence_queue = []

hand_visible_counter = 0
MIN_REQUIRED_FRAMES = 30

prev_time = 0.0
waiting_time = 0.0

refined_sentence = ""         # API가 다듬은 최종 문장
is_refining = False           # API 호출 중 여부 (중복 호출 방지)

# 한글 폰트
try:
    font = ImageFont.truetype("malgun.ttf", 30)
except IOError:
    font = ImageFont.load_default()

# mediapipe 초기화
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

# 모델 파일 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "sign_language_model_v3.pth"
checkpoint = torch.load(save_path, map_location=device)

# 정보 추출
loaded_word_to_idx = checkpoint['word_to_idx']
idx_to_word = {v: k for k, v in loaded_word_to_idx.items()}
hidden_size = checkpoint['hidden_dim']
num_layers = checkpoint['num_layers']
output_dim = len(loaded_word_to_idx)

# 모델 정의
class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SignLanguageClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout= 0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        combined = torch.cat((max_pool, max_pool), dim=1)

        return self.fc(combined)

model = SignLanguageClassifier(feature_dim, hidden_size, output_dim, num_layers).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def refine_sentence(words):
    global refined_sentence, is_refining
    try:
        client = OpenAI(
            api_key="",
            base_url="https://api.groq.com/openai/v1"
        )
        word_str = " ".join(words)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 빠른 모델 추천
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"다음은 한국어 수어 인식 시스템이 순서대로 감지한 단어들이야: [{word_str}]\n"
                        "이 단어들을 자연스러운 한국어 문장 하나로 다듬어줘. "
                        "단어의 순서와 의미를 최대한 유지하고, 문장 외에 다른 설명은 하지 마."
                    )
                }
            ]
        )
        refined_sentence = response.choices[0].message.content.strip()
        print(f"다듬어진 문장: {refined_sentence}", flush=True)
    except Exception as e:
        print(f"API 오류: {e}", flush=True)
        refined_sentence = " ".join(words)
    finally:
        is_refining = False

# 웹캠 캡처
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("카메라가 성공적으로 켜졌습니다.", flush=True)
    print("종료하려면 카메라 창 클릭 후 q를 누르세요.", flush=True)
    time.sleep(2)
    prev_time = time.time()
else:
    print("카메라 연결 실패")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("카메라 화면을 불러올 수 없습니다.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(img_rgb)
    results_pose = pose.process(img_rgb)
    
    left_hand_data = [0.0] * 63
    right_hand_data = [0.0] * 63

    #양손 좌표 추출
    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            hand_label = handedness.classification[0].label

            temp_coords = []
            for lm in hand_landmarks.landmark:
                temp_coords.extend([lm.x, lm.y, 1.0])
            
            if hand_label == "Left":
                left_hand_data = temp_coords
            else:
                right_hand_data = temp_coords

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    
    frame_keypoints = left_hand_data + right_hand_data

    # 코, 양쪽 어깨 좌표 추출
    extra_features = []
    
    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        extra_features.extend([nose.x, nose.y, 1])

        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        extra_features.extend([right_shoulder.x, right_shoulder.y, 1, left_shoulder.x, left_shoulder.y, 1])

        h, w, c = frame.shape
        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 8, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 8, (0, 0, 255), cv2.FILLED)
    else:
        extra_features = [0, 0, 1, 0, 0, 1, 0, 0, 1]

    frame = cv2.flip(frame, 1)

    frame_keypoints = frame_keypoints + extra_features

    # 코 좌표 기준 상대 좌표 계산
    ref_x, ref_y = extra_features[0], extra_features[1]

    x_indices = list(range(0, feature_dim, 3))
    y_indices = list(range(1, feature_dim, 3))
    frame_keypoints = np.array(frame_keypoints, dtype=np.float32)
    frame_keypoints[x_indices] = (frame_keypoints[x_indices] - ref_x)
    frame_keypoints[y_indices] = (frame_keypoints[y_indices] - ref_y)
    frame_keypoints = frame_keypoints.tolist()

    cur_time = time.time()
    delta_time = cur_time - prev_time
    is_hand_detected = results_hands.multi_hand_landmarks is not None

    if not is_hand_detected:
        frame_buffer.append([0.0] * feature_dim)
        waiting_time += delta_time
        if waiting_time >= 3.0:
            word_sequence_queue = []
    else:
        waiting_time = 0.0
        
        frame_buffer.append(frame_keypoints)

    current_buffer = np.array(frame_buffer, dtype=np.float32)
    hand_detected_per_frame = np.sum(np.abs(current_buffer[:, :126]), axis=1) > 0
    actual_hand_frames = np.sum(hand_detected_per_frame)
    print(f"현재 윈도우 내부: 진짜 손이 인식된 프레임은 총 {actual_hand_frames}개 / 175개")

    if actual_hand_frames > MIN_REQUIRED_FRAMES:
        input_window = np.array([frame_buffer], dtype=np.float32)
        input_tensor = torch.tensor(input_window, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        predict = np.argmax(prob)
        conf = prob[predict]            

        if conf > THRESHOLD:
            top3 = np.argsort(prob)[::-1][:5]
            for idx in top3:
                print(f"{idx_to_word[idx]}: {prob[idx]*100:.1f}%")
            detected_word = idx_to_word[predict]

            if not word_sequence_queue or word_sequence_queue[-1] != detected_word:
                word_sequence_queue.append(detected_word)
                print(f"인식 단어: {detected_word} (확률: {conf*100:.1f}%)")

                # 새 단어가 추가될 때마다 백그라운드에서 문장 다듬기 호출
                if not is_refining:
                    is_refining = True
                    refined_sentence = "문장 생성 중..."
                    t = threading.Thread(target=refine_sentence, args=(list(word_sequence_queue),), daemon=True)
                    t.start()

    if word_sequence_queue:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        sentence_text = f"인식된 문장: {' '.join(word_sequence_queue)}"
        draw.text((30, 50), sentence_text, font=font, fill=(255, 0, 0))

        if refined_sentence:
            draw.text((30, 90), f"문장: {refined_sentence}", font=font, fill=(255, 0, 0))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    prev_time = cur_time
    cv2.imshow("Su-eo Project: Hand & Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()