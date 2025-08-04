import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import math
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import pickle


# ========================
# Dataset & DataLoader
# ========================

def create_global_label_mapping(label_paths):
    all_labels = set()
    for path in label_paths:
        y = pd.read_csv(path).iloc[:, 0].astype(int).tolist()
        all_labels.update(y)
    unique_labels = sorted(all_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Global Label Mapping: {label_mapping}")
    return label_mapping

class EEGSequenceDataset(Dataset):
    def __init__(self, X_file, y_file, sequence_length=1, label_mapping=None):
        self.sequence_length = sequence_length
        self.X = pd.read_csv(X_file)
        y_raw = pd.read_csv(y_file).iloc[:, 0].astype(int).values.flatten()

        if label_mapping is None:
            raise ValueError("label_mapping must be provided to ensure consistent label encoding across datasets.")
        self.label_mapping = label_mapping

        valid_indices = [i for i, label in enumerate(y_raw) if label in label_mapping]
        if len(valid_indices) < len(y_raw):
            print(f"Filtered out {len(y_raw) - len(valid_indices)} labels not in mapping.")

        self.X = self.X.iloc[valid_indices].reset_index(drop=True)
        self.y = [label_mapping[y_raw[i]] for i in valid_indices]

        import collections
        label_counts = collections.Counter(self.y)
        print(f"Label distribution in {y_file}: {dict(label_counts)}")

        self.structured_features = []
        for _, row in self.X.iterrows():
            image = []
            for i in range(5):
                coef = row.iloc[i]
                params = row.iloc[5 + 4*i : 5 + 4*(i+1)]
                image.append([coef] + params.tolist())
            self.structured_features.append(image)

        self.image_size = (5, 5)

        if len(self.structured_features) < self.sequence_length:
            raise ValueError(f"Not enough samples for sequence_length={self.sequence_length}. Only {len(self.structured_features)} available.")

    def __len__(self):
        return len(self.structured_features) - self.sequence_length + 1

    def __getitem__(self, idx):
        X_seq = self.structured_features[idx: idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        seq_tensors = [torch.tensor(sample, dtype=torch.float32).unsqueeze(0) for sample in X_seq]
        X_tensor = torch.stack(seq_tensors, dim=0)
        return X_tensor, torch.tensor(y_target, dtype=torch.long)

def compute_class_weights(labels, label_mapping):
    label_counts = Counter(labels)
    total = sum(label_counts[label] for label in label_mapping.keys() if label_counts[label] > 0)
    weights = [total / label_counts[label] if label_counts[label] > 0 else 0.0 for label in label_mapping.keys()]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights

def create_dataloaders(csv_paths, sequence_length=4, batch_size=32, use_residue=False, save_prefix="combined_data"):
    all_features = []
    all_labels = []

    def parse_list(x):
        return ast.literal_eval(x)

    for path in csv_paths:
        df = pd.read_csv(path)
        df["Params"] = df["Params"].apply(parse_list)
        df["Coeffs"] = df["Coeffs"].apply(parse_list)
        if use_residue:
            df["Residue"] = df["Residue"].apply(parse_list)

        for _, row in df.iterrows():
            params = row["Params"]
            coeffs = row["Coeffs"]
            if use_residue:
                residue = row["Residue"]
                feat = coeffs + params + residue
            else:
                feat = coeffs + params
            all_features.append(feat)
            all_labels.append(int(row["label"]))

    X = pd.DataFrame(all_features)
    y_raw = pd.Series(all_labels)

    unique_labels = sorted(y_raw.unique())
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y = y_raw.map(label_mapping)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print("Label mapping:", label_mapping)
    print("Train:", Counter(y_train))
    print("Val:  ", Counter(y_val))
    print("Test: ", Counter(y_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(f"{save_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(X_train_scaled).to_csv(f"{save_prefix}_X_train.csv", index=False)
    y_train.to_csv(f"{save_prefix}_y_train.csv", index=False)
    pd.DataFrame(X_val_scaled).to_csv(f"{save_prefix}_X_val.csv", index=False)
    y_val.to_csv(f"{save_prefix}_y_val.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{save_prefix}_X_test.csv", index=False)
    y_test.to_csv(f"{save_prefix}_y_test.csv", index=False)

    train_dataset = EEGSequenceDataset(f"{save_prefix}_X_train.csv", f"{save_prefix}_y_train.csv", sequence_length, label_mapping)
    val_dataset   = EEGSequenceDataset(f"{save_prefix}_X_val.csv", f"{save_prefix}_y_val.csv", sequence_length, label_mapping)
    test_dataset  = EEGSequenceDataset(f"{save_prefix}_X_test.csv", f"{save_prefix}_y_test.csv", sequence_length, label_mapping)

    class_weights = compute_class_weights(y_train.values, label_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(label_mapping), class_weights, y_train, y_val, y_test

# ========================
# Model Architecture
# ========================

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, kernel_size, 
                 gru_hidden_size, gru_layers, num_classes, cnn_feature_dim):
        """
        A CNN + GRU model adapted for 5x5 image inputs.

        Parameters:
            input_channels: Number of channels in the input (1 for grayscale).
            cnn_out_channels: Number of filters for the CNN layer.
            kernel_size: Convolution kernel size.
            gru_hidden_size: Hidden size for the GRU.
            gru_layers: Number of GRU layers.
            num_classes: Number of output classes.
            cnn_feature_dim: Flattened feature dimension after the CNN block.
                           For a 5x5 input without pooling: cnn_feature_dim = cnn_out_channels * 5 * 5.
        """
        super(CNN_GRU_Model, self).__init__()
        # A lightweight CNN for a 5x5 input.
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
            # No pooling, since 5x5 is already small.
        )
        # GRU to process the sequence of CNN features.
        self.gru = nn.GRU(input_size=cnn_feature_dim, hidden_size=gru_hidden_size,
                          num_layers=gru_layers, batch_first=True, dropout=0.3)
        # Fully connected layer for classification.
        self.fc = nn.Linear(gru_hidden_size, num_classes)
    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []
        for t in range(seq_len):
            cnn_out = self.cnn(x[:, t, :, :, :])  # (batch_size, cnn_out_channels, 5, 5)
            cnn_out = cnn_out.view(batch_size, -1)  # flatten to (batch_size, cnn_feature_dim)
            cnn_features.append(cnn_out)
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, cnn_feature_dim)
        gru_out, _ = self.gru(cnn_features)  # (batch_size, seq_len, gru_hidden_size)
        final_feature = gru_out[:, -1, :]    # last time-step output
        output = self.fc(final_feature)
        return output

# ========================
# Training & Evaluation Functions
# ========================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)  # (batch_size, num_classes)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print("Predictions:", predicted.tolist())
    print("Targets:", y_batch.tolist())
    return epoch_loss, epoch_acc

def test_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    return all_preds, all_targets

# ========================
# Main Training Loop
# ========================

def main_training(train_loader, val_loader, test_loader, num_epochs=20, lr=0.001, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Apply class weights if provided
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)  # shape: (batch_size, num_classes)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

            # 💡 Print softmax predictions for first 3 samples in this batch
            for i in range(min(3, len(X_batch))):
                probs = torch.softmax(outputs[i], dim=0).detach().cpu().numpy()
                # print(f"Softmax[{i}]: {probs.round(3)} | True: {y_batch[i].item()}")

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")

    # Plot training curves
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, [acc*100 for acc in train_accuracies], label="Train Acc")
    plt.plot(epochs, [acc*100 for acc in val_accuracies], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Final test evaluation
    preds, targets = test_model(model, test_loader, device)
    test_acc = sum(1 for p, t in zip(preds, targets) if p == t) / len(targets)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))
    print("Classification Report:")
    print(classification_report(targets, preds))



csv_paths = [
    "act_results_with_labels_sub1.csv",
    "act_results_with_labels_sub2.csv",
    "act_results_with_labels_sub3.csv"
]


train_loader, val_loader, test_loader, num_classes, class_weights, y_train, y_val, y_test = create_dataloaders(
    csv_paths=csv_paths,
    sequence_length=4,
    batch_size=8,
    use_residue=False,
    save_prefix="combined_data"
)


# Instantiate the model with the dynamically determined number of classes.
input_channels = 1           # grayscale image
cnn_out_channels = 16        # modest number of filters
kernel_size = 3              # 3x3 kernel
gru_hidden_size = 64
gru_layers = 2

# For a 5x5 input with no pooling: cnn_feature_dim = cnn_out_channels * 5 * 5.
cnn_feature_dim = cnn_out_channels * 5 * 5

model = CNN_GRU_Model(input_channels, cnn_out_channels, kernel_size, 
                      gru_hidden_size, gru_layers, num_classes, cnn_feature_dim)



unique_labels = np.array(sorted(set(y_train)))
weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train)

class_weights = torch.tensor(weights, dtype=torch.float)

# Run training and evaluation
main_training(train_loader, val_loader, test_loader, num_epochs=10, lr=0.001, class_weights=class_weights)
