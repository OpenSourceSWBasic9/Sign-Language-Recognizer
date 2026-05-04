# 수어 인식기

## 1. 가상환경 생성
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
* [한국어 수어 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=103)
* 가입 후, 파일 목록 들어가서 다운로드 버튼 클릭하여 데이터 신청 후 다운로드 가능 (리눅스 환경 권장)
* `find '[폴더경로]' -name [파일명.zip.part*] -print0 | sort -zt'.' -k2V | xargs -0 cat > [파일명.zip]`
* 압축파일 용량이 크기 때문에 분할되어 있음 → 하나로 합친 후 압축 해제
* `unzip [파일명.zip]`
* unzip 설치 안 되어 있으면 설치할 것. unzip 설치 명령어 `sudo apt update && sudo apt install unzip`

### 주피터 노트북 실행
* `jupyter notebook`
* 위 명령어 실행 시, 링크 생성 → 브라우저에 붙여넣기
* trian.ipynb 실행
* 왜 주피터 노트북을 통해서 모델 학습을 진행하나요?
  * 셀마다 코드를 실행할 수 있기 때문에 한 번 데이터를 불러온 뒤 학습만 하거나, 모델을 불러와서 학습하기 수월함.
