import torch
import torch.nn as nn
from .backbone import get_backbone_model
from .classifier_heads import AnimalClassifier, PigBreedClassifier

class EfficientNetMultiHead(nn.Module):
    def __init__(self, 
                 num_animal_classes=10,     
                 num_pig_breed_classes=5,   
                 backbone_type='efficientnet_b0',  
                 pretrained=True,           
                 **kwargs):      # classifier_type, classifier_kwargs 제거
    
        super().__init__()
        print(f"모델 생성중")
        print(f"  - 찾은 동물 클래스: {num_animal_classes}개")
        print(f"  - 찾은 돼지 품종: {num_pig_breed_classes}개")
        print(f"  - backbone: {backbone_type}")
        
        # 1단계: 백본 모델 로드 
        self.backbone, self.feature_dim = get_backbone_model(
            model_type=backbone_type, 
            pretrained=pretrained 
        )
        print(f"전이학습 모델 로드됨")
        
        print("동물 분류기 생성중")
        self.animal_classifier = AnimalClassifier(
            in_features=self.feature_dim,
            num_classes=num_animal_classes
        )
        
        print("품종 분류기 생성중")
        self.pig_breed_classifier = PigBreedClassifier(
            in_features=self.feature_dim,
            num_classes=num_pig_breed_classes
        )
        
        print(f"모델 완성됨. backbone : {backbone_type} + AnimalClassifier + PigBreedClassifier")
    
    def forward(self, x):
        features = self.backbone(x)
        animal_out = self.animal_classifier(features)
        pig_breed_out = self.pig_breed_classifier(features)
        return animal_out, pig_breed_out
