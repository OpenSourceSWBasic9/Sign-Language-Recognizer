# 수어 인식기

## 1. 가상환경 생성
* 왜 가상환경을 생성하나요?
  * 리눅스가 자체적으로 파이썬을 사용하기 때문에 OS의 핵심 기능이 특정 버전의 라이브러리에 의존
  * 가상환경을 생성하지 않으면 라이브러리 버전을 덮어씌워 오류 발생할 수 있음
  * 다른 프로젝트와 격리
### 가상환경 생성을 위한 패키지 설치
* `sudo apt update`  
* `sudo apt install python3-venv python3-full`  

### 프로젝트 폴더로 이동
* `cd ~/working_directory`

### 'venv'라는 이름의 가상환경 생성
* `python3 -m venv venv`

### 가상환경 활성화
* `source venv/bin/activate`  
* 매번 이 명령어 실행해서 가상환경 활성화해야 함.

### 가상환경 내 pip 업데이트
* `pip install --upgrade pip`  

### 패키지 설치
* `pip install -r requirements.txt`  

## 2. 모델 학습 코드 실행
### 데이터 다운로드
* [한국어 수어 데이터](https://drive.google.com/file/d/1ooQm75JyEElfrCKla5dZ9BUya9By7gN9/view?usp=drive_link)
* 현실 영상 기반 좌표 데이터, 가상 영상 기반 좌표 데이터, 정답 데이터
* AIHub의 한국어 수어 데이터 중 WORD0001부터 WORD0550까지의 범위에서 무작위로 100개 선별

### 주피터 노트북 실행
* `jupyter notebook`
* 위 명령어 실행 시, 링크 생성 → 브라우저에 붙여넣기
* trian.ipynb 실행
* 왜 주피터 노트북을 통해서 모델 학습을 진행하나요?
  * 셀마다 코드를 실행할 수 있기 때문에 한 번 데이터를 불러온 뒤 학습만 하거나, 모델을 불러와서 학습하기 수월함.
