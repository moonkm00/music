import torch
from torchvision import models, transforms
from PIL import Image

class CustomVibeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        numFtrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(numFtrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

def testPrediction():
    model = CustomVibeModel()
    model.load_state_dict(torch.load('custom_vibe_model.pth', map_location='cpu'))
    model.eval()
    
    img = Image.open('oasis_images/images/Acorns 1.jpg').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inputTensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(inputTensor)[0].tolist()
        
    print(f"Acorns 1.jpg -> Predicted Valence: {prediction[0]:.4f}, Predicted Energy: {prediction[1]:.4f}")

if __name__ == "__main__":
    testPrediction()
