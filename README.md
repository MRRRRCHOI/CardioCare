# CardioCare

심장병 예측을 위한 End-to-End 머신러닝 시스템

## 실행 방법

### 1. 저장소 복제

git clone <repository_url>
cd CardioCare


### 2. 의존성 설치


pip install -r requirements.txt

### 3. 모델 학습

python src/train.py

### 4. Docker 이미지 빌드

```bash
docker build -t cardiocare:1.0 .
```

### 5. 테스트 실행

```bash
python -m unittest
```

또는

```bash
pytest
