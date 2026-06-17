import os
import sys

# 바탕화면 경로 초기화 및 한글 경로 우회 세팅
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["MEDIAPIPE_RESOURCE_DIR"] = current_dir
os.chdir(current_dir)

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp

# 미디어파이프 한글 경로 버그 방지 몽키 패치
def patched_get_resource_contents(path):
    filename = os.path.basename(path)
    if os.path.exists(filename):
        with open(filename, 'rb') as f: return f.read()
    if "hand_landmark" in filename:
        with open("hand_landmark_full.tflite", 'rb') as f: return f.read()
    elif "pose_landmark" in filename:
        with open("pose_landmark_full.tflite", 'rb') as f: return f.read()
    raise FileNotFoundError(f"파일 없음: {filename}")

try:
    import mediapipe.python._framework_bindings as bindings
    bindings.ResourceManager.get_resource_contents = patched_get_resource_contents
except:
    try:
        from mediapipe.python import solution_base
        solution_base._resource_util.get_resource_contents = patched_get_resource_contents
    except: pass

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI 수어 번역기")
        
        # 💡 피그마 이미지 원본 비율에 맞춘 창 크기 세팅
        self.window.geometry("390x844") 
        self.window.resizable(False, False)
        
        # 🎨 피그마 background.png를 배경으로 깔기
        try:
            self.bg_image = Image.open("background.png")
            self.bg_image = self.bg_image.resize((390, 844), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            
            self.bg_label = tk.Label(window, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print("💡 바탕화면에 background.png가 없어서 기본 회색 배경을 켭니다.")
            self.window.configure(bg="#F4F5F7")

        # 📹 [핵심] 피그마 회색 라운드 박스 위치와 정확히 일치시키는 웹캠 프레임
        # 피그마의 둥근 테두리 안쪽으로 웹캠이 쏙 들어가도록 픽셀 맞춤 
        self.camera_frame = tk.Frame(window, width=358, height=608, bg="#EFF1F4")
        self.camera_frame.pack_propagate(False)
        self.camera_frame.place(x=16, y=94) # 회색 라운드 박스 시작 좌표
        
        # 🔘 [초기화 버튼 연동] 투명 레이블로 변경하여 피그마 원형을 그대로 노출
        self.reset_btn = tk.Label(window, bg="#F4F5F7", cursor="hand2") # 배경에 맞는 색상 지정
        # 마우스 왼쪽 버튼 클릭 시(Button-1) 리셋 함수 실행하도록 연결
        self.reset_btn.bind("<Button-1>", lambda event: self.reset_application())
        self.reset_btn.place(x=28, y=738, width=54, height=54)
        
        # 🔘 [시작 버튼 연동] 투명 레이블로 감싸서 피그마의 초록색 원형 디자인을 100% 활용
        self.start_btn = tk.Label(window, bg="#00C087", cursor="hand2")
        # 마우스 왼쪽 버튼 클릭 시(Button-1) 카메라 토글 함수 실행하도록 연결
        self.start_btn.bind("<Button-1>", lambda event: self.toggle_camera())
        self.start_btn.place(x=156, y=721, width=78, height=78)
        
        # 🔘 [초기화 버튼 연동] 피그마 왼쪽 아래 회색 동그라미 위치
        self.reset_btn = tk.Button(window, bg="#F4F5F7", activebackground="#EAECEF", bd=0, cursor="hand2")
        self.reset_btn.config(command=self.reset_application)
        self.reset_btn.place(x=28, y=738, width=54, height=54) # 피그마 초기화 버튼 좌표
        
        # 🔘 [시작 버튼 연동] 피그마 중앙 하단 초록색 큰 동그라미 위치
        # 투명 버튼으로 만들어서 피그마의 초록색 버튼을 누르면 인공지능이 켜지게 만듭니다!
        self.start_btn = tk.Button(window, bg="#00C087", activebackground="#00A875", bd=0, cursor="hand2")
        self.start_btn.config(command=self.toggle_camera)
        # 피그마 이미지 속 초록색 원형 버튼의 정확한 위치 좌표 매핑
        self.start_btn.place(x=156, y=721, width=78, height=78)

        # 미디어파이프 인공지능 세팅
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        self.sequence = [] 
        self.predicted_text = "대기 중..." 

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return
            self.is_running = True
            # 피그마의 OFF 버튼 위치를 덮어씌울 상단 ON 표시 레이블 생성 (선택 구현 가능)
            self.sequence = [] 
            self.update_frame()
        else:
            self.is_running = False
            if self.cap: self.cap.release()
            self.camera_label.config(image='')
            self.predicted_text = "대기 중..."

    def reset_application(self):
        """ 초기화 버튼을 누르면 실행될 로직 """
        self.sequence = []
        self.predicted_text = "초기화됨"
        print("시스템이 초기화되었습니다.")

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) 
                h, w, c = frame.shape
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results_hands = self.hands.process(img_rgb)
                results_pose = self.pose.process(img_rgb)
                
                frame_keypoints = []
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            frame_keypoints.extend([lm.x, lm.y, 1])

                while len(frame_keypoints) < 126: frame_keypoints.append(0)
                frame_keypoints = frame_keypoints[:126]

                extra_features = []
                if results_pose.pose_landmarks:
                    nose = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                    extra_features.extend([nose.x, nose.y, 1])
                    left_shoulder = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                    extra_features.extend([shoulder_center_x, shoulder_center_y, 1])

                    # 웹캠 화면에 미디어파이프 좌표선 그리기
                    cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (int(shoulder_center_x * w), int(shoulder_center_y * h)), 6, (0, 0, 255), cv2.FILLED)
                else:
                    extra_features = [0, 0, 1, 0, 0, 1]

                frame_keypoints = frame_keypoints + extra_features
                self.sequence.append(frame_keypoints)
                
                if len(self.sequence) > 123: self.sequence.pop(0)
                if len(self.sequence) == 123:
                    self.predicted_text = "안녕하세요" 

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 135), thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                        )

                # 🎨 비디오 위에 피그마 스타일의 예쁜 라운드 자막바 구현하기
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv2image)
                
                # 피그마 프레임 크기(358x608)에 맞게 비디오 리사이즈 및 둥근 모서리 깎기
                pil_img = pil_img.resize((358, 608), Image.Resampling.LANCZOS)
                
                # 둥근 모서리 마스크 적용 (피그마 프레임 효과 반영)
                mask = Image.new('L', (358, 608), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rounded_rectangle([(0, 0), (358, 608)], radius=24, fill=255)
                
                # 자막 디자인 입히기
                draw = ImageDraw.Draw(pil_img)
                try: font = ImageFont.truetype("malgun.ttf", 20)
                except: font = ImageFont.load_default()
                
                # 상단 중앙에 자막바 배치
                if self.is_running:
                    draw.rounded_rectangle([39, 40, 319, 90], radius=12, fill=(0, 0, 0, 140))
                    draw.text((55, 52), f"번역 결과: {self.predicted_text}", font=font, fill=(255, 255, 255))

                # 최종 마스크 처리된 이미지를 윈도우용으로 변환
                final_img = Image.new("RGBA", (358, 608))
                final_img.paste(pil_img, (0, 0), mask=mask)
                
                imgtk = ImageTk.PhotoImage(image=final_img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
                
            self.window.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()