import os
import sys
import json
import numpy as np
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #photoDataProcessor
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}



    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))


    test_dataset = ImageFolderWithName(root="/root/autodl-tmp/dataset/skinDisease_split/test",
                                        transform=data_transform["val"])
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
    sta = torch.load('/root/code_file/MedMamba/MultiMedmambaLargeNet.pth')
    net.load_state_dict(sta)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params}")
    net.to(device)

    # test
    net.eval()
    acc = 0.0

    num_classes = 6
    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    auroc = MulticlassAUROC(num_classes=num_classes).to(device)
    specificity_metric = MulticlassSpecificity(num_classes=num_classes, average=None).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            #test
            test_images, test_labels,test_filename = test_data
            test_text = np.full((test_images.shape[0], X_test.shape[1]), 0.0, dtype=np.float32)
            for i in range(test_images.shape[0]):
                j = test_filename[i]
                test_text[i] = X_test[test_name_to_idx[j]]
            test_text = torch.from_numpy(test_text).float()
            outputs = net(test_images.to(device),test_text.to(device))
            #loss
            loss = criterion(outputs.to(device), test_labels.to(device))
            running_loss += loss.item()
            test_labels = test_labels.to(device)

            outputs = torch.softmax(outputs, dim=1).to(device)
            predict_y = torch.max(outputs, dim=1)[1].to(device)

            auroc.update(outputs, test_labels)
            accuracy.update(predict_y, test_labels)
            precision.update(predict_y, test_labels)
            recall.update(predict_y, test_labels)
            f1_score.update(predict_y, test_labels)
            specificity_metric.update(predict_y, test_labels)
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    test_accurate = acc / test_num


    specificity_per_class = specificity_metric.compute()
    epoch_auc = auroc.compute()
    prec = precision.compute()
    rec = recall.compute()
    f1 = f1_score.compute()
    print('Overall Accuracy: %.3f' %(test_accurate))
    print(f"AUC: {epoch_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    cnt=0.0
    for i, specificity in enumerate(specificity_per_class):
        print(f"Specificity for Class {i}: {specificity:.4f}")
        cnt+=specificity
    print(f"Average Specificity: {cnt/6:.4f}")
    print("-" * 50)

    print('Finished Testing')


if __name__ == '__main__':
    main()
