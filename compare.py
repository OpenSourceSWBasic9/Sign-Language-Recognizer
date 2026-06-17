import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from collections import deque
from numpy.lib.format import open_memmap

import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── 설정 ──────────────────────────────────────────────
TARGET_WORD = input("비교할 단어를 입력하세요 (훈련 데이터에 있는 단어): ").strip()
MAX_FRAME = 60
feature_dim = 135
x_indices = list(range(0, feature_dim, 3))
y_indices = list(range(1, feature_dim, 3))
WIDTH, HEIGHT = 1920.0, 1080.0

is_recording = False
post_motion_counter = 0

# ── 훈련 데이터 로드 ───────────────────────────────────
print("훈련 데이터 로드 중...")
data = np.load("./Train/raw_sign_dataset.npz", allow_pickle=True)
X_raw = data['X_raw']
y = data['y']
word_to_idx = data['word_to_idx'][0]
idx_to_word = {v: k for k, v in word_to_idx.items()}

if TARGET_WORD not in word_to_idx:
    print(f"'{TARGET_WORD}'가 훈련 데이터에 없습니다.")
    print(f"사용 가능한 단어: {list(word_to_idx.keys())}")
    exit()

# 해당 단어 샘플 하나 꺼내서 process_and_pad
target_idx = word_to_idx[TARGET_WORD]
sample_indices = np.where(y == target_idx)[0]
sample = X_raw[sample_indices[0]]  # 첫 번째 샘플

padded = np.zeros((MAX_FRAME, feature_dim), dtype=np.float32)
actual_len = min(len(sample), MAX_FRAME)
start_idx = MAX_FRAME - actual_len
padded[start_idx:] = sample[:actual_len]

for f in range(start_idx, MAX_FRAME):
    ref_x = padded[f, 126]
    ref_y = padded[f, 127]
    if ref_x != 0 and ref_y != 0:
        padded[f, x_indices] = (padded[f, x_indices] - ref_x) / WIDTH
        padded[f, y_indices] = (padded[f, y_indices] - ref_y) / HEIGHT

train_sample = padded  # (175, 135)
print(f"훈련 샘플 로드 완료: '{TARGET_WORD}' ({actual_len}프레임)")

# ── 실시간 MediaPipe 수집 ──────────────────────────────
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
frame_buffer = deque([[0.0] * feature_dim] * MAX_FRAME, maxlen=MAX_FRAME)

print(f"\n카메라 창에서 '{TARGET_WORD}' 수어를 보여주세요.")
print("스페이스바: 현재 버퍼 캡처 & 비교 / Q: 종료")

captured = False
rt_sample = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(img_rgb)
    results_pose = pose.process(img_rgb)

    left_hand_data = [0.0] * 63
    right_hand_data = [0.0] * 63

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
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 1. 왼손이 검출되지 않았을 때 (모두 0.0일 때)
    if left_hand_data == [0.0] * 63:
        print("왼손 미검출")
        if results_pose.pose_landmarks:
            # 왼쪽 어깨(LEFT_SHOULDER) 좌표를 기준으로 삼음
            ls = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            # 왼쪽 어깨보다 X축으로 살짝 바깥쪽, Y축으로는 허벅지 높이(어깨 아래로 약 +0.4~0.5 정도)
            # 미디어파이프 이미지 좌표계는 아래로 갈수록 Y가 커지므로 +를 해줍니다.
            virtual_left_x = ls.x + 0.05  # 몸 바깥쪽
            virtual_left_y = ls.y + 1.2  # 허벅지/골반 높이
            
            # 왼손의 21개 관절 전체를 이 가상의 차렷 자세 좌표로 채워버림
            temp_virtual = []
            for _ in range(21):
                temp_virtual.extend([virtual_left_x, virtual_left_y, 1.0])
            left_hand_data = temp_virtual

    # 2. 오른손이 검출되지 않았을 때 (필요하다면 오른손 수어 안 할 때를 위해 추가)
    if right_hand_data == [0.0] * 63:
        if results_pose.pose_landmarks:
            rs = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            virtual_right_x = rs.x + 0.05
            virtual_right_y = rs.y + 1.2
            
            temp_virtual = []
            for _ in range(21):
                temp_virtual.extend([virtual_right_x, virtual_right_y, 1.0])
            right_hand_data = temp_virtual

    extra_features = []
    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        extra_features.extend([nose.x, nose.y, 1.0])
        ls = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        extra_features.extend([rs.x, rs.y, 1.0, ls.x, ls.y, 1.0])
    else:
        extra_features = [0.0] * 9

    frame_keypoints = left_hand_data + right_hand_data + extra_features
    frame_keypoints = np.array(frame_keypoints, dtype=np.float32)

    ref_x, ref_y = extra_features[0], extra_features[1]
    if ref_x != 0 and ref_y != 0:
        frame_keypoints[x_indices] = (frame_keypoints[x_indices] - ref_x)
        frame_keypoints[y_indices] = (frame_keypoints[y_indices] - ref_y)
    
    right_hand_x_indices = list(range(63, 126, 3))
    frame_keypoints[right_hand_x_indices] = frame_keypoints[right_hand_x_indices] * -1.0

    target_shoulder_dist = 0.18
    current_shoulder_dist = np.sqrt((ls.x-rs.x)**2+(ls.y-rs.y)**2)
    scale_factor = current_shoulder_dist / target_shoulder_dist

    frame_keypoints[:126] = frame_keypoints[:126] / scale_factor
    print(frame_keypoints[0], frame_keypoints[1])

    is_hand = results_hands.multi_hand_landmarks is not None
    if is_hand:
        frame_buffer.append(frame_keypoints.tolist())
    else:
        frame_buffer.append([0.0] * feature_dim)

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

            if actual_hand_frames >= 20:
                rt_sample = np.array(frame_buffer, dtype=np.float32)
                for f in range(1, len(rt_sample)):
                    # 이번 프레임에서 순간적으로 손을 놓쳐서 0이 되었거나, 
                    # 직전 프레임과의 차이가 비정상적으로 클 때 (칼날 노이즈 감지)
                    for idx in range(0, 126): # 손 영역 좌표들 스캔
                        if rt_sample[f-1, idx] != 0:

                            if rt_sample[f, idx] == 0:
                                rt_sample[f, idx] = rt_sample[f-1, idx]

                            elif np.abs(rt_sample[f, idx] - rt_sample[f-1, idx]) > 0.15:
                                # 직전 프레임의 정상적인 값을 그대로 복사해서 메워버림 (보간 처리)
                                rt_sample[f, idx] = rt_sample[f-1, idx]
                break


    hand_str = "손 감지 O" if is_hand else "손 감지 X"
    cv2.putText(frame, f"{TARGET_WORD} 수어를 보여주세요 | {hand_str}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE: 캡처 & 비교  |  Q: 종료", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("MediaPipe 실시간", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        rt_sample = np.array(frame_buffer, dtype=np.float32)

        for f in range(1, len(rt_sample)):
            # 이번 프레임에서 순간적으로 손을 놓쳐서 0이 되었거나, 
            # 직전 프레임과의 차이가 비정상적으로 클 때 (칼날 노이즈 감지)
            for idx in range(0, 126): # 손 영역 좌표들 스캔
                if rt_sample[f, idx] == 0 or np.abs(rt_sample[f, idx] - rt_sample[f-1, idx]) > 0.15:
                    # 직전 프레임의 정상적인 값을 그대로 복사해서 메워버림 (보간 처리)
                    rt_sample[f, idx] = rt_sample[f-1, idx]

        print("캡처 완료! 비교 그래프 생성 중...")
        break

cap.release()
cv2.destroyAllWindows()

if rt_sample is None:
    print("캡처된 데이터 없음.")
    exit()

# ── 비교 시각화 ────────────────────────────────────────
# 왼손 손목(0,1), 오른손 손목(63,64), 코(126,127) 비교
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f"'{TARGET_WORD}' - 훈련(OpenPose) vs 실시간(MediaPipe) 좌표 비교", fontsize=14)

labels = ["왼손 손목 X(0)", "왼손 손목 Y(1)",
          "오른손 손목 X(63)", "오른손 손목 Y(64)",
          "코 X(126)", "코 Y(127)"]
indices = [0, 1, 63, 64, 126, 127]

for ax, idx, label in zip(axes.flat, indices, labels):
    ax.plot(train_sample[:, idx], label="훈련(OpenPose)", alpha=0.8)
    ax.plot(rt_sample[:, idx], label="실시간(MediaPipe)", alpha=0.8)
    ax.set_title(label)
    ax.legend()
    ax.set_xlabel("프레임")
    ax.set_ylabel("정규화 좌표값")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("compare_result.png", dpi=150, bbox_inches='tight')
plt.show()
print("비교 그래프가 compare_result.png로 저장됐어.")

# 수치 요약
print("\n=== 수치 요약 ===")
for idx, label in zip(indices, labels):
    t_vals = train_sample[train_sample[:, idx] != 0, idx]
    r_vals = rt_sample[rt_sample[:, idx] != 0, idx]
    if len(t_vals) > 0 and len(r_vals) > 0:
        print(f"{label}: 훈련 평균={t_vals.mean():.4f} / 실시간 평균={r_vals.mean():.4f} | "
              f"훈련 std={t_vals.std():.4f} / 실시간 std={r_vals.std():.4f}")