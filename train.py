import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

# ==========================================
# 글로벌 상수 정의 (Global Constants)
# ==========================================
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CSV_PATH = "oasis_images/OASIS.csv"
IMG_DIR = "oasis_images/images"
OUTPUT_CSV = "dataset.csv"
MODEL_SAVE_PATH = "custom_vibe_model.pth"

# ==========================================
# [1] 데이터셋 생성 및 전처리 모듈
# ==========================================
def prepareOasisDataset() -> bool:
    """
    [전처리 ③단계: 실시간 데이터 정합성 검증 및 파일 필터링]
    CSV 메타데이터 파일 내의 Theme 항목과 실제 물리적인 고화질 이미지 디렉토리 파일이 
    일치하여 온전히 연동되는 유효 레코드만 실시간 검증(os.path.exists)하여 선별합니다.
    """
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMG_DIR):
        return False
    dfOasis = pd.read_csv(CSV_PATH)
    records = []
    for _, row in dfOasis.iterrows():
        theme = row["Theme"]
        
        # [전처리 ①단계: 감성 지표 Min-Max 스케일링 (수치 전처리)]
        # OASIS의 1.0~7.0 실수형 정서 지표를 출력층인 Sigmoid 스펙에 맞추어 0.0~1.0 척도로 정규화합니다.
        valNorm = (float(row["Valence_mean"]) - 1.0) / 6.0
        arousalNorm = (float(row["Arousal_mean"]) - 1.0) / 6.0
        
        imgPath = f"oasis_images/images/{theme}.jpg"
        
        # 물리 이미지 파일의 존재성을 정밀 대조 스캔하여 FileNotFoundError 사전 원천 봉쇄
        if os.path.exists(imgPath):
            records.append({"image_path": imgPath, "valence": valNorm, "energy": arousalNorm})
            
    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
    print(f"[Data Prep] Parsed {len(records)} real images successfully with advanced preprocessing!")
    return True

def generateDummyData() -> None:
    """
    [전처리 ④단계: 예외 상황 대비 견고한 데이터 모사 (Robust Fallback)]
    실제 OASIS 데이터 디렉토리가 부재한 배포 및 테스트 상황에서도 시스템이 안전하게 작동하도록,
    지정된 색상별 감성 수치(행복/슬픔/차분/아늑)에 맞춰 모사 이미지를 자동 생성하여 공급합니다.
    """
    print("[Data Prep] Real OASIS dataset not found. Generating dummy preprocessed data...")
    os.makedirs("dummy_dataset", exist_ok=True)
    
    # 색상별 수치 정화 매핑 데이터 풀 (RGB, Valence, Energy)
    dummyColors = [
        ("happy_red", (255, 50, 50), 0.90, 0.90),      # 행복한 빨강 (Happy Red)
        ("sad_blue", (30, 50, 150), 0.20, 0.25),       # 슬픈 파랑 (Sad Blue)
        ("calm_green", (50, 150, 80), 0.80, 0.35),     # 차분한 초록 (Calm Green)
        ("cozy_yellow", (255, 230, 100), 0.95, 0.65)   # 포근한 노랑 (Cozy Yellow)
    ]
    records = []
    for name, rgb, v, e in dummyColors:
        path = f"dummy_dataset/{name}.jpg"
        Image.new("RGB", (224, 224), rgb).save(path)
        records.append({"image_path": path, "valence": v, "energy": e})
    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
    print("[Data Prep] Generated dummy dataset with robust preprocessing fallback successfully!")

# ==========================================
# [2] PyTorch 커스텀 데이터셋 로더 클래스
# ==========================================
class VibeDataset(Dataset):
    def __init__(self, csvFile: str, transform=None):
        self.dataFrame = pd.read_csv(csvFile)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataFrame)

    def __getitem__(self, idx: int):
        imgPath = self.dataFrame.iloc[idx, 0]
        image = Image.open(imgPath).convert("RGB")
        target = torch.tensor([
            float(self.dataFrame.iloc[idx, 1]),
            float(self.dataFrame.iloc[idx, 2])
        ], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target

# ==========================================
# [3] AI 감성 회귀 모델 정의
# ==========================================
class CustomVibeModel(nn.Module):
    def __init__(self):
        super(CustomVibeModel, self).__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.resnet50(pretrained=True)
            
        numFtrs = self.backbone.fc.in_features
        # 출력층 설계 (Regression Head)
        self.backbone.fc = nn.Sequential(
            nn.Linear(numFtrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# ==========================================
# [4] 고속 특징 추출 및 학습 프로세스
# ==========================================
def runTraining() -> None:
    print("=" * 60)
    print("Starting Lightning-Fast VibeFrame Custom Mood AI Training...")
    print("=" * 60)
    
    if not prepareOasisDataset():
        generateDummyData()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: [{device.type.upper()}] (PyTorch CUDA is_available: {torch.cuda.is_available()})")
    
    # 1. 피처 추출용 모델 준비 (FC 레이어를 Identity로 대체하여 2048차원 피처 획득)
    print("Loading pre-trained ResNet50 for feature extraction...")
    try:
        extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except AttributeError:
        extractor = models.resnet50(pretrained=True)
        
    for param in extractor.parameters():
        param.requires_grad = False
    
    # 마지막 레이어를 항등원(Identity)으로 설정하여 피처만 출력되게 함
    extractor.fc = nn.Identity()
    extractor = extractor.to(device)
    extractor.eval()
    
    # 2. 피처 추출 데이터셋 로드 및 화상 전처리 파이프라인
    # [전처리 ②단계: 시각 이미지 텐서 변환 및 ImageNet 표준 규격 정규화 (화상 전처리)]
    preprocess = transforms.Compose([
        # [Resize] 고해상도 이미지를 Backbone 수용 영역 규격인 224x224 픽셀로 강제 표준화
        transforms.Resize((224, 224)),
        # [ToTensor] HWC 형식을 CHW 형식 텐서로 변환하고 픽셀 실수(0.0~1.0) 자동 스케일링
        transforms.ToTensor(),
        # [Normalize] ImageNet 도메인의 평균값 및 표준편차를 적용하여 경사 소실 방지 및 고속 수렴 촉진
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rawDataset = VibeDataset(OUTPUT_CSV, transform=preprocess)
    rawLoader = DataLoader(rawDataset, batch_size=32, shuffle=False)
    
    print("Extracting stable ResNet50 features (Only 1 Pass)...")
    allFeatures = []
    allTargets = []
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(rawLoader):
            images = images.to(device)
            feats = extractor(images)
            allFeatures.append(feats.cpu())
            allTargets.append(targets)
            if (idx + 1) % 10 == 0:
                print(f"Extracted features for { (idx + 1) * 32 } images...")
                
    featuresTensor = torch.cat(allFeatures, dim=0)
    targetsTensor = torch.cat(allTargets, dim=0)
    print(f"Extraction complete! Features Shape: {featuresTensor.shape}, Targets Shape: {targetsTensor.shape}")
    
    # 3. 고속 MLP 학습 데이터로더 생성
    mlpDataset = TensorDataset(featuresTensor, targetsTensor)
    mlpLoader = DataLoader(mlpDataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # MLP 헤드 설계
    class MLPHead(nn.Module):
        def __init__(self, inputDim=2048):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(inputDim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.fc(x)
            
    mlp = MLPHead().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    
    print(f"Training Regressor MLP Head for {NUM_EPOCHS} epochs directly on cached features...")
    for epoch in range(NUM_EPOCHS):
        mlp.train()
        runningLoss = 0.0
        for feats, targets in mlpLoader:
            feats = feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = mlp(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * feats.size(0)
            
        epochLoss = runningLoss / len(mlpDataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d}] --- Loss (MSE): {epochLoss:.5f}")
            
    # 4. 학습된 MLP 가중치를 최종 CustomVibeModel에 복사 및 저장
    print("Re-assembling complete CustomVibeModel...")
    finalModel = CustomVibeModel()
    
    # MLP 가중치를 finalModel.backbone.fc에 완벽 이식
    finalModel.backbone.fc.load_state_dict(mlp.fc.state_dict())
    
    # 최종 모델 저장
    torch.save(finalModel.state_dict(), MODEL_SAVE_PATH)
    print(f"🎯 Successfully saved high-diversity improved model to '{MODEL_SAVE_PATH}'!")

if __name__ == "__main__":
    runTraining()
