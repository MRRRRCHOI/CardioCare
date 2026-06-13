# CardioCare

심장병 예측을 위한 End-to-End 머신러닝 프로젝트입니다.

## 프로젝트 구조

├── data/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md

## 실행 방법

### 1. 저장소 복제

```bash
git clone <repository-url>
cd CardioCare
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 모델 학습

```bash
python src/train.py
```

### 4. Docker 이미지 빌드

```bash
docker build -t cardiocare:1.0 .
```

### 5. 테스트 실행

pytest 사용 시:

```bash
pytest
```

또는 unittest 사용 시:

```bash
python -m unittest discover tests
```

## 사용 기술

- Python
- pandas
- numpy
- scikit-learn
- MLflow
- Docker
- unittest
- GitHub Actions