""" Visualization with t-SNE. 
"""

import os
import argparse
from tqdm import tqdm
import torch
from torch.amp import autocast
import timm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from datasets import setup_val_dataloader
from util.utils import load_model


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/GenImage2', help='Root of dataset. ')
    parser.add_argument('--num_samples_each_class', type=int, default=1024,  help='Number of samples in each class. ')
    parser.add_argument('--ckpt_path', type=str, default="output_dir/ckpt/resnet50_200000.pth", help='Checkpoint path. ')
    parser.add_argument('--test_class', type=str, default="ADM", help='Test class. ')
    args = parser.parse_args()

    IMAGE_FOLDERS = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]

    val_dataloaders = {
        folder: setup_val_dataloader(
            folder_path=os.path.join(args.data_root, folder, "val"), 
            batch_size=32, 
            num_workers=8, 
        ) for folder in IMAGE_FOLDERS
    }

    model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
    load_model(args.ckpt_path, model=model)

    model.fc = torch.nn.Identity()
    model.cuda()
    model.eval()

    data_list = []
    label_list = []
    with torch.no_grad(): 
        for i, folder in enumerate(tqdm(IMAGE_FOLDERS)): 
            num = 0
            for batch, _ in val_dataloaders[folder]: 
                batch = batch.cuda()

                with autocast(enabled=True, device_type="cuda"):
                    outputs = model(batch)
                data_list.append(outputs.cpu())
                label_list.append(torch.tensor([i] * len(outputs)))
                num += len(outputs)
                if num >= args.num_samples_each_class: 
                    break
    
    data = torch.cat(data_list, dim=0)
    label = torch.cat(label_list)

    data = data.numpy()

    tsne = TSNE(n_components=2, init="pca")
    X = tsne.fit_transform(data)
    Y = label.numpy()

    color_list = [
        '#D2D0D1', 
        '#AF7AC5', 
        '#5DADE2', 
        '#F4D03F', 
        '#EC7063', 
        '#58D68D', 
        '#F5B041', 
    ]

    IMAGE_FOLDERS2 = ["Real", "ADM", "BigGAN", "GLIDE", "MJ", "SD", "VQDM"]
    for i in range(0, len(IMAGE_FOLDERS2)): 
        if IMAGE_FOLDERS2[i] == args.test_class: 
            label_name = IMAGE_FOLDERS2[i] + " (test)"
        else: 
            label_name = IMAGE_FOLDERS2[i]
        dataX = X[i * args.num_samples_each_class: (i + 1) * args.num_samples_each_class]
        plt.scatter(dataX[:, 0], dataX[:, 1], c=color_list[i], s=3, label = label_name)
    plt.legend(loc="upper left", title="Classes")
    plt.show()


if __name__ == "__main__": 
    main()

