import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ==========================================
# 설정 및 하이퍼파라미터
# ==========================================
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu") # MPS 오류 발생 시 CPU 강제 사용

print("==========================================")
print("   꽃 데이터 분류 과제 (PyTorch 버전)")
print(f"   사용 디바이스: {DEVICE}")
print("==========================================\n")

# ==========================================
# 1. 데이터셋 클래스 정의
# ==========================================
class FlowerDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), label

def load_data_paths(folder_path):
    file_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    
    print(f"[Info] 클래스 목록: {class_names}")
    
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        files = os.listdir(class_path)
        count = 0
        for f in files:
            if f.startswith('.'): continue
            file_paths.append(os.path.join(class_path, f))
            labels.append(i)
            count += 1
        print(f"  - {class_name}: {count}장")
        
    return np.array(file_paths), np.array(labels), class_names

# ==========================================
# 2. 데이터 준비
# ==========================================
train_dir = './archive/train'
file_paths, labels, class_names = load_data_paths(train_dir)

# 학습/검증 데이터 분할
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(), # 0~1로 정규화됨
])

train_dataset = FlowerDataset(train_paths, train_labels, transform=transform)
val_dataset = FlowerDataset(val_paths, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n[Info] 학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개 준비 완료.\n")

# ==========================================
# 3. 모델 정의
# ==========================================

# 과제 1: CNN 모델
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv(32) -> MaxPool
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv(64) -> MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv(128) -> MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Flatten size 계산: 128 -> 64 -> 32 -> 16. 128채널 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 과제 2: ResNet50
def get_resnet50(num_classes):
    # weights='IMAGENET1K_V1' 대신 weights='DEFAULT' 사용 가능
    print("[Info] ResNet50 가중치 다운로드 중... (시간이 조금 걸릴 수 있습니다)")
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # 마지막 레이어 교체
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model

# ==========================================
# 4. 학습 함수
# ==========================================
def train_model(model, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    print(f">>> [{model_name}] 학습 시작...")
    
    for epoch in range(EPOCHS):
        # 학습
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        history['accuracy'].append(epoch_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # 최종 검증 데이터에 대한 예측값 수집 (혼동 행렬용)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return history, all_labels, all_preds

def plot_results(history, title):
    epochs_range = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['accuracy'], label='Train Acc')
    plt.plot(epochs_range, history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} Accuracy')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.grid()
    
    filename = f"{title}_result.png"
    plt.savefig(filename)
    print(f"[Info] 그래프 저장 완료: {filename}")
    # plt.show() # 파일 저장만 하고 창은 띄우지 않음 (자동화를 위해)

def plot_confusion_matrix_custom(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{title} Confusion Matrix')
    
    filename = f"{title}_confusion_matrix.png"
    plt.savefig(filename)
    print(f"[Info] 혼동 행렬 저장 완료: {filename}")
    # plt.show()

# ==========================================
# 5. 실행
# ==========================================
num_classes = len(class_names)

# 과제 1 실행
cnn_model = SimpleCNN(num_classes)
cnn_history, cnn_labels, cnn_preds = train_model(cnn_model, "Task1_CNN")
plot_results(cnn_history, "Task1_CNN")
plot_confusion_matrix_custom(cnn_labels, cnn_preds, class_names, "Task1_CNN")

print("\n------------------------------------------\n")

# 과제 2 실행
resnet_model = get_resnet50(num_classes)
resnet_history, resnet_labels, resnet_preds = train_model(resnet_model, "Task2_ResNet50")
plot_results(resnet_history, "Task2_ResNet50")
plot_confusion_matrix_custom(resnet_labels, resnet_preds, class_names, "Task2_ResNet50")

print("\n[Done] 모든 작업이 완료되었습니다.")