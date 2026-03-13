import torch.nn as nn
from torchvision import models


def get_backbone_model(model_type='efficientnet_b0', pretrained=True):
    
    print(f"백본 모델 로딩 중: {model_type}")
    
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier[1].in_features  # 1280
        model.classifier = nn.Identity()  
        
    elif model_type == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier[1].in_features  # 1536
        model.classifier = nn.Identity()
        
    elif model_type == 'resnet50':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        feature_dim = model.fc.in_features  # 2048
        model.fc = nn.Identity()  
        
    elif model_type == 'vgg16':
        model = models.vgg16(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier[6].in_features  # 4096
        model.classifier = nn.Identity()
        
    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier[3].in_features  # 576
        model.classifier = nn.Identity()
        
    elif model_type == 'densenet121':
        model = models.densenet121(weights='DEFAULT' if pretrained else None)
        feature_dim = model.classifier.in_features  # 1024
        model.classifier = nn.Identity()
        
    else:
        raise ValueError(f"지원하지 않는 백본 모델: {model_type}")
    
    print(f"백본 준비 완료! {model_type} (특징 {feature_dim})")
    return model, feature_dim

