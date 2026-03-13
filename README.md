# 동물 분류 멀티헤드 CNN 프로젝트

**멀티태스크 딥러닝을 활용한 동물 분류 + 돼지 품종 분류 시스템**을 개발했습니다.

입력 이미지를 받아서 **10종류 동물 분류**를 수행하고, 만약 돼지로 판단되면 **5종 돼지 품종 추가 분류**를 진행하는 시스템입니다. 돼지를 타겟으로 선정한 이유는 축산업에서의 품종 분류 실용성과 멀티헤드 학습 구조 검증을 위함입니다.

---

## 프로젝트 개요

### 목표
- **1차 분류**: 10종 동물 구분을 수행했습니다 (bear, cat, dog, dolphin, eagle, elephant, fox, horse, pig, sheep)
- **2차 분류**: 돼지 품종 구분을 구현했습니다 (berkshire, duroc, hampshire, landrace, yorkshire)

- **멀티헤드 구조**를 도입하여 두 작업을 하나의 모델에서 동시에 수행하도록 했습니다.
- **Docker 패키징**을 통해 깔끔한 개발 환경을 구성했습니다.

### 구조적 특징
- **공유 백본**: 하나의 모델로 두 가지 일을 처리하여 효율성을 높였습니다.
- **조건부 학습**: 돼지인 경우에만 품종 분류를 진행하도록 설계했습니다.
- **모듈화**: 각 부품을 쉽게 교체할 수 있도록 구현했습니다.

---

## 기술 스택

### 주요 라이브러리 선택 이유
- **PyTorch** (2.8.0+cu128): Keras와 비교했을 때 더 직관적이고 세부 제어가 용이했으며, CUDA 지원과 Docker 이미지 통합이 편리했습니다.
- **TorchVision** (0.23.0+cu128): Pretrained 모델 사용과 데이터 증강을 위한 다양한 기능을 제공했습니다.
- **FastAPI** (0.115.5): 첫 테스트때 경험해본 적 없다는 이유로 사용하지 않았기 때문에 적용해보았고, ML 모델 서빙에 유리하다는 평이 많아 채택했습니다.
- **Docker**: 개발/배포 환경의 일관성 유지와 버전 관리가 편리했으며, pytorch 베이스 이미지 사용으로 환경 구성이 수월했습니다.

### Docker 설정
```dockerfile
# 베이스 이미지
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# 작업 디렉토리
WORKDIR /app

# Git 설치
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 포트 설정
EXPOSE 8000

# 실행 명령어
CMD ["python", "backend/api/main.py"]
```

---

## 모델 구조

### 백본 모델 선택 이유
EfficientNet-B0를 선택한 주요 이유는 다음과 같습니다:
- 적은 파라미터 수(5.3M)로 효율적인 학습이 가능했습니다.
- ResNet-18(11M)이나 VGG-11(130M+)에 비해 훨씬 가벼웠습니다.
- 작은 프로젝트 규모와 데이터셋에 적합했습니다.
- MBConv 블록 기반으로 빠른 연산이 가능했습니다.

### 핵심 구현 코드
```python
def get_backbone_model(model_type='efficientnet_b0', pretrained=True):
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier[1].in_features  # 1280개 특징
        model.classifier = nn.Identity()  # 분류층 제거
    return model, feature_dim

class BackboneNetMultiHead(nn.Module):
    def __init__(self, num_animal_classes=10, num_pig_breed_classes=5,
                 backbone_type='efficientnet_b0', classifier_type='dropout', 
                 dropout=0.3, hidden_dim=256):
        super().__init__()
        
        # 1. 백본 모델
        self.backbone, self.feature_dim = get_backbone_model(
            model_type=backbone_type, pretrained=True
        )
        
        # 2. 동물 분류기
        self.animal_classifier = create_classifier_head(
            classifier_type=classifier_type,
            in_features=self.feature_dim,
            num_classes=num_animal_classes,
            dropout=dropout, hidden_dim=hidden_dim
        )
        
        # 3. 품종 분류기
        self.pig_breed_classifier = create_classifier_head(
            classifier_type=classifier_type,
            in_features=self.feature_dim, 
            num_classes=num_pig_breed_classes,
            dropout=dropout, hidden_dim=hidden_dim
        )
    
    def forward(self, x):
        features = self.backbone(x)
        animal_out = self.animal_classifier(features)
        pig_breed_out = self.pig_breed_classifier(features)
        return animal_out, pig_breed_out
```

---

## 학습 과정

### 데이터셋 구조
```
dataset/
├── animals/train/     # 동물 분류용 학습 데이터
│   ├── bear/         # ~700장
│   ├── cat/          # ~830장  
│   ├── dog/          # ~720장
│   └── ...           # 총 10종 동물
├── animals/val/       # 동물 분류용 검증 데이터
├── breeds/train/      # 품종 분류용 학습 데이터
│   ├── berkshire/    # ~680장
│   ├── duroc/        # ~610장
│   └── ...           # 총 5종 품종
└── breeds/val/        # 품종 분류용 검증 데이터
```

### 데이터 전처리 및 증강
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # 크기 맞추기
    transforms.RandomResizedCrop((192, 192), scale=(0.9, 1.0)),  # 랜덤 자르기
    transforms.RandomHorizontalFlip(p=0.5),           # 좌우 뒤집기
    transforms.RandomRotation(degrees=10),            # 회전
    transforms.ColorJitter(                           # 색상 변경
        brightness=0.1, contrast=0.1, 
        saturation=0.1, hue=0.05
    ),
    transforms.ToTensor(),                            # 텐서로 변환
    transforms.Normalize(                             # 정규화
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

### 학습 파라미터 설정
```python
config = {
    'batch_size': 64,
    'epochs': 15,  # Early Stopping으로 인해 30에서 15로 감소
    'lr': 0.001,
    'backbone_type': 'efficientnet_b0',
    'classifier_type': 'dropout',
    'classifier_dropout': 0.3
}
```


## 손실함수와 옵티마이저

본 프로젝트의 동물 분류와 돼지 품종 분류는 **다중 클래스 분류** 문제입니다.  
따라서 Cross-Entropy Loss와 Adam 옵티마이저를 사용하였습니다.

### Cross-Entropy Loss
\[
\mathcal{L}_{CE} = - \sum_{c=1}^{C} y_c \log(\hat{y}_c)
\]
- \(C\) : 클래스 수  
- \(y_c\) : 실제 레이블 (원-핫)  
- \(\hat{y}_c\) : 모델 예측 확률 (Softmax 출력)  

### 멀티헤드 구조 손실 합산
\[
\mathcal{L}_{total} = \mathcal{L}_{animal} + \mathbf{1}_{\text{animal=pig}} \cdot \mathcal{L}_{pig\_breed}
\]

### Adam 옵티마이저
\[
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
\]
- 학습률 적응형 업데이트, 빠른 수렴과 안정적 학습 지원



### API 서버 구현
```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        animal_out, pig_out = model(input_tensor)
        
        animal_probs = F.softmax(animal_out, dim=1)
        animal_pred = animal_probs.argmax(dim=1).item()
        animal_confidence = animal_probs[0][animal_pred].item()
        
        result = {
            "animal": ANIMAL_CLASSES[animal_pred],
            "confidence": round(animal_confidence, 3)
        }
        
        if ANIMAL_CLASSES[animal_pred] == "pig":
            pig_probs = F.softmax(pig_out, dim=1)
            pig_pred = pig_probs.argmax(dim=1).item()
            pig_confidence = pig_probs[0][pig_pred].item()
            
            result["breed"] = BREED_CLASSES[pig_pred]
            result["breed_confidence"] = round(pig_confidence, 3)
        
        return result
```

## 모델 성능 및 결과

### 최종 성능
- **동물 분류 정확도**: 85.9%
- **돼지 품종 분류 정확도**: 58.7%
- **추론 속도**: ~200ms (EfficientNet-B0, CPU)
- **모델 크기**: ~20MB (압축 후)
- **지원 이미지**: JPG, PNG (224x224로 자동 리사이즈)
- **학습 환경**: NVIDIA RTX 4060TI(VRAM 8GB), RYZEN 5 7500F(6 코어, 12 스레드)

### 백본 모델별 성능 비교
| 모델 | 파라미터 수 | 메모리 사용량 | 동물 정확도 | 품종 정확도 | 추론 속도 | 특징 |
|------|------------|--------------|------------|------------|-----------|-------|
| EfficientNet-B0 | 5.3M | ~400MB | 85.9% | 58.7% | 200ms | 가장 균형잡힌 성능 |
| EfficientNet-B3 | 12.0M | ~800MB | 87.2% | 60.1% | 280ms | B0보다 성능 향상, 더 큰 메모리 필요 |
| ResNet50 | 25.6M | ~1.2GB | 86.8% | 59.8% | 320ms | 안정적인 성능, 큰 메모리 요구 |
| MobileNet-V3-Small | 2.9M | ~200MB | 82.1% | 55.3% | 150ms | 가장 가벼움, 모바일 환경에 적합 |
| VGG16 | 138M | ~2GB | 86.5% | 59.4% | 480ms | 높은 정확도, 매우 큰 메모리와 느린 속도 |
| DenseNet121 | 8.0M | ~600MB | 86.1% | 58.9% | 250ms | 특징 재사용으로 효율적, 중간 크기 |

### 모델별 학습 곡선
#### EfficientNet-B0
![EfficientNet-B0 학습 곡선](/chart/efficient_b0_plot.png)
- 안정적인 학습 곡선
- 15 에폭에서 조기 종료
- 빠른 초기 수렴과 안정적인 성능 유지

#### EfficientNet-B3
![EfficientNet-B3 학습 곡선](/chart/efficient_b3_plot.png)
- B0보다 높은 최종 성능
- 더 긴 학습 시간 필요
- 메모리 사용량 증가로 배치 크기 조정 필요

#### ResNet50
![ResNet50 학습 곡선](/chart/resnet50_plot.png)
- 초기 수렴이 빠름
- 메모리 사용량 증가
- 안정적인 검증 성능

#### MobileNet-V3-Small
![MobileNet 학습 곡선](/chart/mobilenet_plot.png)
- 가장 빠른 학습 속도
- 상대적으로 낮은 정확도
- 적은 메모리 사용량

#### DenseNet121
![DenseNet121 학습 곡선](/chart/densenet121_plot.png)
- 특징 재활용으로 효율적 학습
- 중간 수준의 성능과 속도
- EfficientNet-B0와 유사한 메모리 요구

#### Chart 종합
![종합 차트](/chart/model_chart.png)


### 모델 선택 고려사항
1. **EfficientNet-B0 선택 이유**:
   - 파라미터 수(5.3M) 대비 높은 성능
   - 적은 메모리 사용량으로 개발 환경에 적합
   - 200ms의 합리적인 추론 속도

2. **다른 모델들의 한계**:
   - ResNet50: 높은 성능이지만 메모리 요구량이 큼
   - VGG16: 가장 높은 정확도지만 파라미터가 너무 많고 느림
   - MobileNet: 가볍지만 정확도가 많이 떨어짐
   - DenseNet121: 좋은 성능이지만 B0 대비 이점이 크지 않음

3. **추가적으로 고려해볼 요소**
- 추가 평가 지표
- 정확도 외에, 클래스 불균형 대응을 위해 **Precision, Recall, F1-score, Confusion Matrix** 사용 가능

### 분류기 타입별 성능 비교
| 분류기 타입 | 파라미터 수 | 동물 정확도 | 품종 정확도 | 특징 |
|------------|------------|------------|------------|-------|
| Simple | ~1.3K | 83.2% | 56.1% | 가장 기본적인 구조 |
| Dropout | ~1.3K | 85.9% | 58.7% | 과적합 방지 효과 |
| MLP | ~300K | 85.1% | 57.9% | 더 복잡한 패턴 학습 |

### 학습 과정 기록
| epoch | train_loss | animal_train_loss | breed_train_loss | animal_val_loss | breed_val_loss | animal_accuracy | breed_accuracy | learning_rate |
|-------|------------|-------------------|------------------|-----------------|----------------|-----------------|----------------|---------------|
| 1     | 2.129     | 0.894             | 1.235           | 0.608          | 1.093          | 0.807          | 0.594          | 0.001         |
| 2     | 1.499     | 0.550             | 0.950           | 0.624          | 1.090          | 0.810          | 0.562          | 0.001         |
| 15    | 0.234     | 0.089             | 0.145           | 0.565          | 1.562          | 0.859          | 0.587          | 0.00035       |

## 겪은 문제점 및 개선 방안

### 1. 로그 출력 부족으로 인한 디버깅 어려움
**문제점**:
- 초기에 로그 출력 명령을 충분히 넣지 않아, 정상 출력이 아닌 경우 어떤 문제가 발생했는지 파악하기 어려웠습니다.
- 예외처리 로그를 구현하지 않아, 에러가 발생한 정확한 분기점을 찾기 힘들었습니다.
- 경로 관련 로그가 없어서 ImportError 발생 시 원인 파악이 어려웠습니다.

**개선 방안**:
```python
# 상세한 로깅 시스템 구현
print(f"백본 모델 로딩 중: {model_type}")
print(f"모델 파일 발견: {model_path}")
print(f"예측 요청: {file.filename}")

# 예외 처리 강화
try:
    model = EfficientNetMultiHead(...)
    print("모델 로드 성공")
except Exception as e:
    print(f"모델 로드 실패: {e}")
```

### 2. 클래스 설계 및 코드 구조 문제
**문제점**:
- 일부 클래스가 제 기능을 하지 못하고 불필요한 호출을 반복하여 연산 효율이 떨어졌습니다.
- 변수명이 불명확하고 중복된 기능의 변수들이 존재했습니다.
- 실제로 사용되지 않는 매개변수들이 있어 코드 구조가 복잡해졌습니다.

**개선 방안**:
- 각 모듈의 역할을 명확히 분리했습니다: backbone.py, classifier_heads.py, multihead_model.py
- 파이썬의 타입 힌트를 활용하여 변수의 의도를 명확히 했습니다.
- 불필요한 매개변수를 제거하고 코드를 정리했습니다.

### 3. 데이터셋 구축 문제
**문제점**:
- 웹 크롤링으로 수집한 데이터에 노이즈가 많았습니다 (관련 없는 이미지, 일러스트 등).
- 실제 학습에 사용할 수 있는 유효 데이터의 수가 매우 적었습니다.
- 클래스별 데이터 수의 차이가 커서 불균형 문제가 발생했습니다.

**개선 방안**:
```python
def _validate_data(self):
    """데이터 분포 검증 및 불균형 경고"""
    max_count = max(animal_counts.values())
    min_count = min(animal_counts.values())
    ratio = max_count / min_count
    if ratio > 3:
        print(f"클래스 불균형 감지! (최대/최소 비율: {ratio:.1f})")
        print("   → 가중치 손실함수 사용을 고려해보세요")
```
> 데이터 불균형 대응 전략

클래스별 데이터 수 차이로 불균형 문제가 존재합니다.  
현재 적용하거나 향후 적용 가능한 대응 방법:
- **Weighted CrossEntropy**: 클래스별 가중치 적용
- **Focal Loss**: 어려운 샘플에 집중
\[
\mathcal{L}_{FL} = - \sum_{c} \alpha_c (1-\hat{y}_c)^\gamma y_c \log(\hat{y}_c)
\]
- **Label Smoothing**: 과적합 방지, 일반화 향상


### 4. 환경 설정 및 배포 지식 부족
**문제점**:
- 환경 설정, PATH 설정, 배치 스크립트 등 개발 환경 구성에 대한 경험이 부족했습니다.
- Docker 환경과 로컬 개발 환경의 차이로 인한 문제가 자주 발생했습니다.

**개선 방안**:
- Docker 이미지에 전체 개발 환경을 패키징하여 일관성을 확보했습니다.
- requirements.txt에 정확한 버전을 명시하여 의존성 문제를 해결했습니다.
- 개발용과 배포용 디렉토리를 분리하여 환경을 깔끔하게 관리했습니다.
- Python venv를 사용하여 로컬 개발 환경을 격리했습니다.

### 5. 머신러닝 기본 지식 부족
**문제점**:
- 과적합과 미적합에 대한 개념은 알고 있었으나, 실제 발생 시 대처가 미숙했습니다.
- 하이퍼파라미터 튜닝 경험이 부족하여 최적값 설정에 어려움을 겪었습니다.
- 학습 과정에서 나오는 다양한 평가 지표들을 제대로 해석하지 못했습니다.

**개선 방안**:
- Early Stopping을 구현하여 과적합을 방지했습니다.
- 학습률 스케줄러를 도입하여 수렴 속도를 개선했습니다.
- 데이터 증강을 통해 학습 데이터의 다양성을 확보했습니다.

## 기술적 배운 점

- **멀티헤드 구조**: 하나의 모델로 여러 작업을 수행하는 효율적인 구조를 실제로 구현하면서 이해할 수 있었습니다.
- **전이 학습**: ImageNet으로 사전 학습된 가중치의 놀라운 성능과 학습 속도를 직접 경험했습니다.
- **데이터 증강**: 제한된 데이터셋에서도 다양한 증강 기법으로 성능을 개선할 수 있다는 것을 배웠습니다.
- **API 설계**: 다른 시스템과의 연동을 위해 필요한 설정들과 구조의 중요성을 이해했습니다.
- **협업의 중요성**: 프론트엔드, 백엔드, 모델링 등 여러 영역을 혼자 다루면서 실제 협업의 가치를 간접적으로 체험했습니다.

## 향후 계획

1. **성능 개선**
   - 데이터 증강 기법 추가 실험
   - 하이퍼파라미터 자동 튜닝

2. **기능 확장**
   - 더 많은 클래스 분류
   - 다른 동물의 품종 분류

---
# 평가 및 제안점

## 평가
1. Vison Deep learning Task에 대하여 모델 학습 방법에 대하여 이해하려고 공부한 결과가 나타남.
2. Multi Head Architecture에 대하여 신규 기법을 도입하였으며 이에 대한 성능 결과를 정량적인 결과를 도출함.
3. 라이브러리 버전 명시와 코드의 모듈화를 통해 관리 및 유지 보수가 쉽도록 정성적인 결과를 도출함.
4. Classification Model에 대하여 이해가 전반적으로 아쉬우나 다양한 모델 활용 및 정량적 평가를 통해 각 모델의 특징을 실험적으로 이해하는 모습을 보임.
5. 학습 및 검증 과정에 있어서 Loss와 Accuracy의 Epoch 당 그래프를 통해 과적합 및 과소적합을 객관적으로 판단할 수 있는 지표를 도출함.
6. Early Stopping 및 Dropout에 대한 내용을 이해하고 있으며 발표 내용에 대한 이해와 질문에 대한 적합한 답변을 구상함.
7. 프로젝트 진행 중 문제점과 이에 대한 해결방안, 추후 성능 향상 방안에 대한 요소를 작성하여 추후 발전 가능성을 보임.

 ## 제안점

1. 다양한 Classificaiton Model 내부의 Layer 및 Architecture에 대한 이해가 필요로 함.
2. 딥러닝 학습 과정에 대한 추가적인 학습을 필요로 함.
3. 딥러닝 학습에 필요한 파라미터에 대한 추가적인 이해를 필요로 함.
4. Classifciation Task에 대한 추가적인 성능평가지표를 탐색하고, Class간 불균형이 나타났을 때 적절한 평가지표를 탐색하는 방안이 필요함.