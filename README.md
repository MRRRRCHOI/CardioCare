# CardioCare

심장병 예측을 위한 End-to-End 머신러닝 시스템

## 실행 방법
1. 저장소 복제

git clone <repository_url>
cd CardioCare

2. 의존성 설치

pip install -r requirements.txt

3. 모델 학습

python src/train.py

4. MLflow 확인

mlflow ui

브라우저에서: http://127.0.0.1:5000

5. 추론 실행

python src/inference.py

6. 테스트 실행

python -m unittest 또는 pytest

7. 드리프트 탐지

python src/monitor.py

8. Docker 실행
docker build -t cardiocare:1.0 .

docker run cardiocare:1.0