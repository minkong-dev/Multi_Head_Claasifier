# 동물 분류 & 돼지 품종 분류

## 컨테이너 실행하기
```bash
docker-compose up --build
```

## 이미지 빌드 (변경사항 업데이트)
```bash
docker build -t ['이미지 이름'] .
```

## 중지
```bash
docker-compose down
```

<!-- 백엔드 API 서버 컨테이너 실행시 자동으로 실행 -->
포트: `http://localhost:8000` (API), 

<!-- HTML 접근 주소 -->
`http://localhost:8000` (index.html)


## 기능
- 동물 10종 분류 (bear, cat, dog, dolphin, eagle, elephant, fox, horse, pig, sheep)
- 돼지면 품종 5개 추가 분류 (berkshire, duroc, hampshire, landrace, yorkshire)

## 모델
- **백본**
    > EfficientNet-B0의 ImageNet pretrained 모델을 사용했으며, 출력 레이어만 제거하여 사용하였습니다.
- **구조**
    > 멀티헤드 구조를 적용해보았습니다. 하나의 모델로 두 가지의 헤드를 도출해 낼 수 있습니다. (동물 + 돼지품종)
- **손실함수**
    > CrossEntropyLoss를 적용해보았습니다. 
- **옵티마이저**
    > Adam (lr=0.001)을 사용하였습니다. 

## 데이터셋
- **준비방법**
    > Google, naver에 Target 검색어를 넣었을 경우의 이미지 검색 결과값 썸네일을 크롤링 했습니다. (github/AutoCrawler)
- **동물**
    > 각 클래스 600~1100장 (train/val 7:3 data_split 스크립트 사용)
- **품종**
    > 각 클래스 300~700장 (train/val 7:3 data_split 스크립트 사용)
- **전처리**
    > Resize(224), Normalize(ImageNet 평균/표준편차), transform 적용
