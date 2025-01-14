import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from model import VSSM as medmamba
from torchmetrics.classification import MulticlassSpecificity,MulticlassAUROC,MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    device = torch.device('cpu')
    #photoDataProcessor
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


    # test_dataset = ImageFolderWithName(root="/root/autodl-tmp/dataset/skinDisease_split/test",
    #                                     transform=data_transform["val"])
    image_path="/root/autodl-tmp/dataset/skinDisease_split/test/MEL/PAT_109_868_723.png"
    test_dataset = Image.open(image_path)
    test_dataset=data_transform["val"](test_dataset)

    test_dataset=test_dataset.unsqueeze(0)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    #textDataProcessor
    df_test = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/test.csv")
    df_train = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/train.csv")
    # df_train.dropna(inplace=True)
    # df_test.dropna(inplace=True)

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

    test_name_to_idx = {s: i for i, s in enumerate(X_test[:,-1])}
    X_test = X_test[:, :-1]
    X_test = X_test.astype(np.float32)

    #init
    input_dim = X_test.shape[1]
    hidden_dims = [128, 64, 32]
    net = medmamba(MLP_input_dim=65, MLP_hidden_dims=hidden_dims, num_classes=6)
    sta = torch.load('/root/code_file/MedMamba/MultiMedmambaLargeNet.pth', map_location=torch.device('cpu'))
    net.load_state_dict(sta)
    net = net.to(device)
    total_params = sum(p.numel() for p in net.parameters())


    # test
    net.eval()
    acc = 0.0

    num_classes = 6
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            #test
            test_images= test_data
            test_filename=os.path.basename(image_path)
            test_images=test_images.to(device)
            test_text = np.full((test_images.shape[0], X_test.shape[1]), 0.0, dtype=np.float32)
            j = test_name_to_idx[test_filename]
            test_text[0] = X_test[j]
            test_text = torch.from_numpy(test_text).float()
            test_text=test_text.to(device)
            outputs = net(test_images,test_text)
            #loss


            outputs = torch.softmax(outputs, dim=1)

            predict_y,prob = torch.max(outputs, dim=1)
            print(predict_y)
            print(prob)







if __name__ == '__main__':
    main()
