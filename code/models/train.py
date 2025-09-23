import json
from datetime import datetime
import mlflow
import torch
import sys
import os
import shutil
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.process_data import load_data, get_transform, size
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

mlflow.set_experiment("PMLDL Assignment 1")

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * (self.input_dim // 8) * (self.input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * (self.input_dim // 8) * (self.input_dim // 8))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def train_model(model, epochs, train_loader, criterion, optimizer, device):
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
    print("Training finished.")
    return model

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print("Evaluation finished.")
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1_score": f1,
    }
    return model, metrics

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(path):
    model = CNN(input_dim=size)
    try:
        model.load_state_dict(torch.load(path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def main():
    # Prefer processed split if available; fall back to raw split-on-the-fly
    processed_train = "data/processed/train"
    processed_test = "data/processed/test"

    if os.path.isdir(processed_train) and os.path.isdir(processed_test):
        train_dataset = load_data(processed_train, get_transform())
        test_dataset = load_data(processed_test, get_transform())
    else:
        print("Error: processed split not found")
        exit(1)

    # Persist labels for API/app consistency
    labels_path = "models/labels.json"
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    labels = getattr(train_dataset, "dataset", train_dataset).classes if hasattr(train_dataset, "dataset") else train_dataset.classes
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # CI overrides
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    train_max = os.environ.get("TRAIN_MAX_SAMPLES")
    test_max = os.environ.get("TEST_MAX_SAMPLES")

    if train_max is not None:
        try:
            n = min(int(train_max), len(train_dataset))
            indices = random.sample(range(len(train_dataset)), n)
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        except Exception:
            pass
    if test_max is not None:
        try:
            n = min(int(test_max), len(test_dataset))
            indices = random.sample(range(len(test_dataset)), n)
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        except Exception:
            pass

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device selection: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = CNN(input_dim=size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = int(os.environ.get("EPOCHS", "20"))
    comparing_metric = "val_f1_score"

    train_model(model, epochs, train_loader, criterion, optimizer, device)
    model, new_metrics = evaluate_model(model, test_loader, criterion, device)
    mlflow.log_metric("val_loss", new_metrics["val_loss"])
    mlflow.log_metric("val_accuracy", new_metrics["val_accuracy"])
    mlflow.log_metric("val_precision", new_metrics["val_precision"])
    mlflow.log_metric("val_recall", new_metrics["val_recall"])
    mlflow.log_metric("val_f1_score", new_metrics["val_f1_score"])

    # Save candidate model first
    os.makedirs("models", exist_ok=True)
    candidate_path = "models/model_candidate.pth"
    torch.save(model.state_dict(), candidate_path)

    prev_model_path = "models/model.pth"
    prev_metrics_path = "models/metrics.json"

    prev_metrics = None
    if os.path.isfile(prev_metrics_path):
        try:
            with open(prev_metrics_path, "r", encoding="utf-8") as f:
                prev_metrics = json.load(f)
        except Exception:
            prev_metrics = None
    elif os.path.isfile(prev_model_path):
        try:
            prev_model = CNN(input_dim=size)
            state = torch.load(prev_model_path, map_location=device)
            prev_model.load_state_dict(state)
            prev_model.to(device)
            _, prev_metrics = evaluate_model(prev_model, test_loader, criterion, device)
        except Exception as e:
            print(f"Error loading previous model: {e}")
            prev_metrics = None

    if prev_metrics is not None:
        prev_score = prev_metrics[comparing_metric]
    else:
        prev_score = 0

    new_score = new_metrics[comparing_metric]

    selected = "new"
    if prev_score is not None and new_score is not None and prev_score >= new_score:
        selected = "previous"

    if selected == "new":
        # Replace best model
        shutil.copy2(candidate_path, prev_model_path)
        best_metrics = new_metrics
    else:
        best_metrics = prev_metrics

    print(f"Previous score: {prev_score}, New score: {new_score}")
    print(f"Selected: {selected}")
    
    # Persist metrics with selection info
    best_metrics_record = {
        **(best_metrics or {}),
        "selected_model": selected,
        "timestamp": datetime.now().isoformat() + "Z",
    }
    with open(prev_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics_record, f, ensure_ascii=False, indent=2)

    # MLflow logging: model and selection decision
    mlflow.log_param("model_selection", selected)
    try:
        mlflow.log_artifact(prev_metrics_path)
    except Exception:
        pass
    if selected == "new":
        mlflow.pytorch.log_model(model, "model")

    # Cleanup candidate file
    try:
        os.remove(candidate_path)
    except Exception:
        pass

    print(model)

if __name__ == "__main__":
    main()