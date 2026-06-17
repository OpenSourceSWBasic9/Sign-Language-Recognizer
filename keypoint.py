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
MAX_FRAME = 60
THRESHOLD = 0.6
TARGET_SHOULDER_DIST = 0.18

frame_buffer = deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME)
real_hand_buffer = deque([[True, True] for _ in range(60)], maxlen=MAX_FRAME)
word_sequence_queue = []

hand_visible_counter = 0
MIN_REQUIRED_FRAMES = 20

prev_time = 0.0
waiting_time = 0.0

refined_sentence = ""         # API가 다듬은 최종 문장
is_refining = False           # API 호출 중 여부 (중복 호출 방지)

is_recording = False
post_motion_counter = 0

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
save_path = "sign_language_model_v4.pth"
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

            refined_sentence = review_response.choices[0].message.content.strip()
            print(f"2차 검수 실행: {refined_sentence}", flush=True)

        else:
            refined_sentence = first_sentence
            print(f"2차 검수 생략: {refined_sentence}", flush=True)
  
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

    frame = cv2.flip(frame, 1)
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
                hand_color = (0,255,0)
            else:
                right_hand_data = temp_coords
                hand_color = (0,0,255)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

    is_left_hand_real = (left_hand_data != [0.0] * 63)
    is_right_hand_real = (right_hand_data != [0.0 * 63])
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


    frame_keypoints = frame_keypoints + extra_features

    # 코 좌표 기준 상대 좌표 계산
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
    real_hand_buffer.append([is_left_hand_real, is_right_hand_real])

    current_buffer = np.array(frame_buffer, dtype=np.float32)
    hand_detected_per_frame = np.sum(np.abs(current_buffer[:, :126]), axis=1) > 0
    actual_hand_frames = np.sum(hand_detected_per_frame)

    xy_indices = [i for i in range(135) if i % 3 != 2]
    pure_xy = current_buffer[-3:, xy_indices]
    frame_diff = np.abs(pure_xy[1:] - pure_xy[:-1])
    motion_amount = np.sum(frame_diff)

    if motion_amount > 6.0 and not is_recording:
        is_recording = True
        post_motion_counter = 0
        print("수어 감지 시작", flush=True)
    
    if is_recording:
        if motion_amount < 0.1:
            post_motion_counter += 1
        else:
            post_motion_counter = 0

        if post_motion_counter >= 5:
            is_recording = False
            post_motion_counter = 0

            if actual_hand_frames >= MIN_REQUIRED_FRAMES:
                input_window = np.array([frame_buffer], dtype=np.float32)
                real_flags = np.array(real_hand_buffer)

                for f in range(1, len(input_window)):
                    # 이번 프레임에서 순간적으로 손을 놓쳐서 0이 되었거나, 
                    # 직전 프레임과의 차이가 비정상적으로 클 때 (칼날 노이즈 감지)
                    for idx in range(0, 126): # 손 영역 좌표들 스캔
                        hand_type = 0 if idx < 63 else 1
                        was_real = real_flags[f-1, hand_type]
                        is_real = real_flags[f, hand_type]

                        if input_window[f-1, idx] != 0:

                            if not was_real and is_real:
                                rt_sample[f, idx] = rt_sample[f-1, idx] * 0.6 + rt_sample[f, idx] * 0.4
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
                        top3 = np.argsort(prob)[::-1][:5]
                        for idx in top3:
                            print(f"{idx_to_word[idx]}: {prob[idx]*100:.1f}%")
                        detected_word = idx_to_word[predict]

                        if not word_sequence_queue or word_sequence_queue[-1] != detected_word:
                            word_sequence_queue.append(detected_word)
                            frame_buffer = deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME)       
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