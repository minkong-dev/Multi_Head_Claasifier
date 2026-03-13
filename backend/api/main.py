import torch
import torch.nn.functional as F
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  
from PIL import Image
from torchvision import transforms
import io
import sys
import os

# 경로 설정 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.multihead_model import EfficientNetMultiHead

# FastAPI 앱 생성
app = FastAPI(
    title="동물 분류 및 돼지 품종 분류", 
    description="동물 종류 추론 후, 돼지라면 품종까지 추론하여 레이블을 반환해요.",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# 클래스 정의 
ANIMAL_CLASSES = ['bear', 'cat', 'dog', 'dolphin', 'eagle', 
                'elephant', 'fox', 'horse', 'pig', 'sheep']
BREED_CLASSES = ['berkshire', 'duroc', 'hampshire', 'landrace', 'yorkshire']

print(f"API 초기화: 동물 {len(ANIMAL_CLASSES)}개, 품종 {len(BREED_CLASSES)}개")

# 전역 변수
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}")

def load_model():
    global model
    print("모델 로딩 시작")
    
    # 가중치 파일
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
    model_path = os.path.join(weights_dir, "final_model.pth")
    
    if os.path.exists(model_path):
        print(f"모델 파일 확인: {model_path}")
        
        # 모델 생성
        model = EfficientNetMultiHead(
            num_animal_classes=10, 
            num_pig_breed_classes=5
        )
        
        # 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"모델을 불러왔어요 : (final_model.pth)")
    else:
        print(f"final_model.pth 모델을 찾지 못했어요. weights 폴더를 확인해주세요 : {weights_dir}")

# 이미지 전처리 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

@app.on_event("startup")
async def startup():
    print("API 서버를 구동할게요")
    load_model()

@app.get("/")
async def root():
    # 405 에러때문에 앱 마운트 경로를 루트에서 /frontend로 바꾸게 되면서
        # 루트 url로 접속시 index.html이 있는 경로로 리다이렉트 설정
    return RedirectResponse(url="/frontend/index.html") 
        

@app.get("/classes")
async def get_classes():
    return {
        "animals": ANIMAL_CLASSES,
        "pig_breeds": BREED_CLASSES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "모델이 로드되지 않았어요"}
    
    print(f"예측 요청: {file.filename}")
    
    try:
        # 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 전처리
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            animal_out, pig_out = model(input_tensor)
            
            # 동물 분류 결과
            animal_probs = F.softmax(animal_out, dim=1)
            animal_pred = animal_probs.argmax(dim=1).item()
            animal_confidence = animal_probs[0][animal_pred].item()
            
            result = {
                "animal": ANIMAL_CLASSES[animal_pred],
                "confidence": round(animal_confidence, 3)
            }
            
            print(f"동물 예측: {result['animal']} ({result['confidence']:.3f})")
            
            # 돼지인 경우 품종도 분류
            if ANIMAL_CLASSES[animal_pred] == "pig":
                pig_probs = F.softmax(pig_out, dim=1)
                pig_pred = pig_probs.argmax(dim=1).item()
                pig_confidence = pig_probs[0][pig_pred].item()
                
                result["breed"] = BREED_CLASSES[pig_pred]
                result["breed_confidence"] = round(pig_confidence, 3)
                
                print(f"품종 예측: {result['breed']} ({result['breed_confidence']:.3f})")
            
            return result
            
    except Exception as e:
        error_msg = f"예측 실패: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

# 배포용 실행
if __name__ == "__main__":
    print("배포 모드로 실행 중...")
    
    try:
        import uvicorn
        print("uvicorn으로 서버 시작...")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            reload=False
        )
    except ImportError:
        print("uvicorn이 설치되지 않았어요. pip install uvicorn 으로 먼저 설치해주세요.")