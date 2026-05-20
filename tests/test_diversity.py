import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

class CustomVibeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.resnet50(pretrained=True)
        numFtrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(numFtrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.backbone(x)

def testDiversity():
    model = CustomVibeModel()
    if os.path.exists("custom_vibe_model.pth"):
        model.load_state_dict(torch.load("custom_vibe_model.pth", map_location=torch.device('cpu')))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imageDir = "oasis_images/images"
    imageFiles = [f for f in os.listdir(imageDir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]

    print("=== Testing Diversity ===")
    for imgName in imageFiles:
        imgPath = os.path.join(imageDir, imgName)
        try:
            pilImg = Image.open(imgPath).convert("RGB")
            tensor = preprocess(pilImg).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor)[0].numpy()
            print(f"{imgName:30s} -> Valence: {pred[0]:.4f}, Energy: {pred[1]:.4f}")
        except Exception as e:
            print(f"Error loading {imgName}: {e}")

if __name__ == "__main__":
    testDiversity()
