import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        self.prev_dim = input_dim
        self.output_dim = output_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(self.prev_dim, hdim))
            layers.append(nn.ReLU())
            self.prev_dim = hdim

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x=self.net(x)
        print(x.shape)
        nn.Linear(self.prev_dim, self.output_dim)
        return x


def  train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch.float())
        loss = criterion(outputs, y_batch.long())  # 如果 y_batch 是 Long 类型整数标签

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch.float())
            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total


def main():
    df_train = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/train.csv")
    df_val = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/val.csv")

    df_train.dropna(inplace=True)
    df_val.dropna(inplace=True)

    df_train = df_train.iloc[:, 2:]
    df_val = df_val.iloc[:, 2:]

    X_train = df_train.drop(["diagnostic", "img_id","background_father","background_mother","elevation"], axis=1)
    y_train = df_train["diagnostic"].values

    X_val = df_val.drop(["diagnostic", "img_id","background_father","background_mother","elevation"],  axis=1)
    y_val = df_val["diagnostic"].values
    X_val.to_csv("/root/code_file/MedMamba/X_val.csv", index=False)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    numeric_features = ["age","diameter_1","diameter_2"]
    categorical_features = ["gender","smoke","drink","region","grew","changed","pesticide", "skin_cancer_history","cancer_history","has_sewage_system","fitspatrick","itch","hurt","bleed","biopsed","has_piped_water"]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore',sparse_output=False)


    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    print(X_train.shape)
    print(X_val.shape)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    print(X_train.shape)
    print(X_val.shape)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)


    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    hidden_dims = [64, 32]
    output_dim = len(np.unique(y_train))

    model = MLP(input_dim, hidden_dims, output_dim).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    epochs = 100
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
