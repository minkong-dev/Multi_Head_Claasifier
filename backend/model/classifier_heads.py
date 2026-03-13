import torch.nn as nn

class AnimalClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        print("AnimalClassifier 사용 중")

    def forward(self, x):
        return self.fc(x)

class PigBreedClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        print("PigBreedClassifier 사용 중")

    def forward(self, x):
        return self.fc(x)
