import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.utils.class_weight import compute_class_weight

class EEGSequenceDataset(Dataset):
    def __init__(self, X_file, y_file, sequence_length=1, label_mapping=None):
        self.sequence_length = sequence_length
        self.X = pd.read_csv(X_file)
        y_raw = pd.read_csv(y_file).iloc[:, 0].astype(int).values

        if label_mapping is None:
            raise ValueError("A label_mapping must be provided.")

        self.label_mapping = label_mapping

        valid_indices = [i for i, label in enumerate(y_raw) if label in self.label_mapping]
        self.X = self.X.iloc[valid_indices].reset_index(drop=True)
        self.y = [self.label_mapping[y_raw[i]] for i in valid_indices]

        self.structured_features = []
        for _, row in self.X.iterrows():
            row_list = row.tolist()
            image = []
            for i in range(5):
                coef = row_list[i]
                params = row_list[5 + 4*i : 5 + 4*(i+1)]
                image.append([coef] + params)
            self.structured_features.append(image)

        if len(self.structured_features) < self.sequence_length:
            raise ValueError(
                f"Not enough samples for sequence_length={self.sequence_length}. "
                f"Only {len(self.structured_features)} available."
            )

    def __len__(self):
        return len(self.structured_features) - self.sequence_length + 1

    def __getitem__(self, idx):
        X_seq = self.structured_features[idx : idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]

        seq_tensors = [
            torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            for sample in X_seq
        ]
        X_tensor = torch.stack(seq_tensors, dim=0)
        return X_tensor, torch.tensor(y_target, dtype=torch.long)

def oversample_dataframe(X, y):
    df = X.copy()
    df['label'] = y
    counts = df['label'].value_counts()
    max_count = counts.max()
    df_list = []
    for cls in counts.index:
        df_cls = df[df['label'] == cls]
        df_cls_oversampled = df_cls.sample(max_count, replace=True, random_state=42)
        df_list.append(df_cls_oversampled)
    df_oversampled = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
    y_oversampled = df_oversampled['label']
    X_oversampled = df_oversampled.drop(columns=['label'])
    return X_oversampled, y_oversampled

def create_dataloaders(csv_paths, sequence_length=4, batch_size=32, save_prefix="combined_data"):
    all_features = []
    all_labels = []

    def parse_list(x):
        return ast.literal_eval(x)

    for path in csv_paths:
        df = pd.read_csv(path)
        df["Params"] = df["Params"].apply(parse_list)
        df["Coeffs"] = df["Coeffs"].apply(parse_list)
        for _, row in df.iterrows():
            params = row["Params"]
            coeffs = row["Coeffs"]
            raw_label = int(row["label"])
            feat = coeffs + params
            all_features.append(feat)
            all_labels.append(raw_label)

    X_all = pd.DataFrame(all_features)
    y_all = pd.Series(all_labels)

    drop_set = {-2, 8, 3}
    mask = ~y_all.isin(drop_set)
    X_all = X_all[mask].reset_index(drop=True)
    y_all = y_all[mask].reset_index(drop=True)

    unique_remaining = sorted(y_all.unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_remaining)}
    print("Final label_mapping:", label_mapping)

    y_mapped = y_all.map(label_mapping)
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_mapped, test_size=0.3, stratify=y_mapped, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print("Training distribution before oversample:", Counter(y_train))
    X_train, y_train = oversample_dataframe(X_train, y_train)
    print("Training distribution after oversample:", Counter(y_train))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(f"{save_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(X_train_scaled).to_csv(f"{save_prefix}_X_train.csv", index=False)
    y_train.map({v: k for k, v in label_mapping.items()}).to_csv(f"{save_prefix}_y_train.csv", index=False)
    pd.DataFrame(X_val_scaled).to_csv(f"{save_prefix}_X_val.csv", index=False)
    y_val.map({v: k for k, v in label_mapping.items()}).to_csv(f"{save_prefix}_y_val.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{save_prefix}_X_test.csv", index=False)
    y_test.map({v: k for k, v in label_mapping.items()}).to_csv(f"{save_prefix}_y_test.csv", index=False)

    train_dataset = EEGSequenceDataset(f"{save_prefix}_X_train.csv", f"{save_prefix}_y_train.csv", sequence_length, label_mapping=label_mapping)
    val_dataset = EEGSequenceDataset(f"{save_prefix}_X_val.csv", f"{save_prefix}_y_val.csv", sequence_length, label_mapping=label_mapping)
    test_dataset = EEGSequenceDataset(f"{save_prefix}_X_test.csv", f"{save_prefix}_y_test.csv", sequence_length, label_mapping=label_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(label_mapping), label_mapping



# ========================
# 4) CNN + GRU Model
# ========================
class CNN_GRU_Model(nn.Module):
    """
    Simple CNN + GRU architecture for 5x5 images (extracted from each 25-D vector).
    """
    def __init__(self, input_channels, cnn_out_channels, kernel_size, 
                 gru_hidden_size, gru_layers, num_classes, cnn_feature_dim):
        super(CNN_GRU_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []
        # Process each time-step individually
        for t in range(seq_len):
            out = self.cnn(x[:, t, :, :, :])  # shape: [batch, cnn_out_channels, 5, 5]
            out = out.view(batch_size, -1)
            cnn_features.append(out)
        # Stack into [batch_size, seq_len, cnn_feature_dim]
        cnn_features = torch.stack(cnn_features, dim=1)
        gru_out, _ = self.gru(cnn_features)
        final_feature = gru_out[:, -1, :]
        output = self.fc(final_feature)
        return output


# ========================
# 5) Training & Evaluation
# ========================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    return running_loss / total, correct / total

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
    return running_loss / total, correct / total

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

def main_training(train_loader, val_loader, test_loader, num_epochs=20, lr=0.001, class_weights=None):
    """
    Main training loop: trains the model, prints epoch stats, then evaluates on test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if class_weights is not None:
        print(f"Using class weights on device {device}: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")

    # Final test evaluation
    preds, targets = test_model(model, test_loader, device)
    test_acc = np.mean([p == t for p, t in zip(preds, targets)])
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))
    print("Classification Report:")
    print(classification_report(targets, preds))

# ========================
# 6) Main Script
# ========================
if __name__ == "__main__":
    csv_paths = [
        "act_results_with_labels_sub1.csv",
        "act_results_with_labels_sub2.csv",
        "act_results_with_labels_sub3.csv"
    ]

    # Create Dataloaders: forcibly drop any label in {-2, 8, 3}, leaving only {0, 1, 2, 4}
    train_loader, val_loader, test_loader, num_classes, label_mapping = create_dataloaders(
        csv_paths=csv_paths,
        sequence_length=4,
        batch_size=8,
        save_prefix="combined_data"  # or any prefix you like
    )

    # Build the CNN+GRU model
    input_channels = 1
    cnn_out_channels = 16
    kernel_size = 3
    gru_hidden_size = 64
    gru_layers = 2
    cnn_feature_dim = cnn_out_channels * 5 * 5  # (16 out_channels) * (5 * 5 spatial)

    model = CNN_GRU_Model(
        input_channels,
        cnn_out_channels,
        kernel_size,
        gru_hidden_size,
        gru_layers,
        num_classes,
        cnn_feature_dim
    )

    # Optionally compute class weights for balanced training
    train_labels = np.array(train_loader.dataset.y)
    weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_labels
    )
    weights_tensor = torch.tensor(weights_np, dtype=torch.float)
    print("Final class weights:", weights_tensor)

    # Train
    main_training(
        train_loader,
        val_loader,
        test_loader,
        num_epochs=100,
        lr=0.001,
        class_weights=weights_tensor
    )
