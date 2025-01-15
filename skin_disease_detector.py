import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
from mamba_model import VSSM as medmamba
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pre_CNN_model import CNN


class ImageFolderWithName(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        filename = os.path.basename(path)
        return sample, target, filename


def main():
    image_path = "/root/autodl-tmp/preModel/train/normal/3321234.jpg"
    df_test = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/test.csv")
    df_train = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/train.csv")
    sta = torch.load('/root/code_file/MedMamba/MultiMedmambaLargeNet.pth', map_location=torch.device('cpu'))
    pre_sta = torch.load('/root/code_file/MedMamba/preCNN.pth',map_location=torch.device('cpu'))

    device = torch.device('cpu')
    # photoDataProcessor
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    test_dataset = Image.open(image_path).convert("RGB")
    test_dataset = data_transform["val"](test_dataset)
    test_dataset = test_dataset.unsqueeze(0)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    #text process
    df_test = df_test.iloc[:, 2:]
    df_train = df_train.iloc[:, 2:]

    X_train = df_train.drop(["diagnostic", "background_father", "background_mother", "elevation"], axis=1)
    X_test = df_test.drop(["diagnostic", "background_father", "background_mother", "elevation"], axis=1)

    numeric_features = ["age", "diameter_1", "diameter_2"]
    categorical_features = ["gender", "smoke", "drink", "region", "grew", "changed", "pesticide", "skin_cancer_history",
                            "cancer_history", "has_sewage_system", "fitspatrick", "itch", "hurt", "bleed", "biopsed",
                            "has_piped_water"]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    test_name_to_idx = {s: i for i, s in enumerate(X_test[:, -1])}
    X_test = X_test[:, :-1]
    X_test = X_test.astype(np.float32)

    # init
    input_dim = X_test.shape[1]
    hidden_dims = [128, 64, 32]
    net = medmamba(MLP_input_dim=65, MLP_hidden_dims=hidden_dims, num_classes=6)

    net.load_state_dict(sta)
    net = net.to(device)
    total_params = sum(p.numel() for p in net.parameters())
    classes = {
        0: "ACK",
        1: "BCC",
        2: "MEL",
        3: "NEV",
        4: "SCC",
        5: "SEK"
    }
    # test
    net.eval()
    acc = 0.0
    num_classes = 6

    pre_model = CNN().to(device)
    pre_model.load_state_dict(pre_sta)
    pre_model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            # test
            test_images = test_data
            test_images = test_images.to(device)

            pre_output = pre_model(test_images)
            predicted = (pre_output > 0.5).float()
            if predicted.sum() == 0:
                test_filename = os.path.basename(image_path)
                test_text = np.full((test_images.shape[0], X_test.shape[1]), 0.0, dtype=np.float32)
                j = test_name_to_idx[test_filename]
                test_text[0] = X_test[j]
                test_text = torch.from_numpy(test_text).float()
                test_text = test_text.to(device)
                outputs = net(test_images, test_text)
                outputs = torch.softmax(outputs, dim=1)
                prob, predict_y = torch.max(outputs, dim=1)
                print('You have a %.3f%% chance of having a %s disease.' % (prob.sum() * (1 - pre_output.sum()) * 100, classes[predict_y.item()]))
            else:
                print('You have a %.3f%% chance of not having a disease.' % (pre_output.sum() * 100))


if __name__ == '__main__':
    main()
