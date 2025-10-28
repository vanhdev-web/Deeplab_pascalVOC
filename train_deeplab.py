import argparse
import cv2
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import os

class VocDataset(VOCSegmentation):
    def __init__(self, root, year, image_set ,download, transform= None, target_transform = None):
        super().__init__(root, year, image_set ,download, transform, target_transform)
        self.classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        self.transform = transform

    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        label = np.array(label, np.int64)
        label[label == 255] = 0
        return image, label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path","-d", type = str, default=r"C:\Users\dovie\OneDrive\Desktop\vietanh\deeplearning\Pascal_VOC\dataset\archive")
    parser.add_argument("--image_size","-imgsz", type = int, default=224)
    parser.add_argument("--epoch","-e", type = int, default=100)
    parser.add_argument("--batch_size","-b", type = int, default=4)
    parser.add_argument("--logging_folder","-l", type = str, default="tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default=None)
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")




    args = parser.parse_args()
    return args

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(args.logging_folder):
        shutil.rmtree(args.logging_folder)
        os.mkdir(args.logging_folder)
    else:
        os.mkdir(args.logging_folder)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)



    root = args.data_path
    image_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])

    ])
    target_transform =  Compose([
        Resize((args.image_size, args.image_size)),

    ])
    train_dataset = VocDataset(root = root, download= False, year="2012", image_set="train",transform=image_transform, target_transform= target_transform)
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=False
    )


    test_dataset = VocDataset(root = root, download= False, year="2012", image_set="val",transform=image_transform, target_transform= target_transform)
    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False
    )

    num_epochs = args.epoch
    num_iters = len(train_dataloader)
    writer = SummaryWriter(args.logging_folder)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)

    best_acc = 0
    acc = MulticlassAccuracy(len(train_dataset.classes)).to(device)
    jaccard = MulticlassJaccardIndex(len(train_dataset.classes)).to(device)
    all_losses = []

    if args.checkpoint_folder:
        checkpoint = torch.load(args.checkpoint_folder)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        epoch = 0

    print(os.path.join(args.trained_models, "best.pt"))

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output["out"], labels)
            all_losses.append(loss.item())
            avg_loss = np.mean(all_losses)
            writer.add_scalar("Train/Loss",avg_loss,epoch * num_iters + iter)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description("Epoch {}/{} Loss {:.3f}".format(epoch + 1, num_epochs, loss))
        model.eval()
        with torch.no_grad():
            all_acc = []
            all_jaccard = []
            progress_bar = tqdm(test_dataloader, colour="yellow")
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output["out"], labels)
                progress_bar.set_description("Epoch {}/{} Loss {:.3f}".format(epoch, num_epochs, loss))

                acc_value = acc(output["out"], labels).item()
                jaccard_value = jaccard(output["out"], labels).item()
                all_acc.append(acc_value)
                all_jaccard.append(jaccard_value)
            avg_acc = np.mean([acc.to("cpu") for acc in all_acc])
            avg_jaccard = np.mean([jacc.to("cpu") for jacc in all_jaccard])
            writer.add_scalar("Val/Acurracy",avg_acc,epoch )
            writer.add_scalar("Val/Jaccard", avg_jaccard, epoch)
            print(avg_acc)
            print(avg_jaccard)

            if avg_acc > best_acc:
                print("model update")
                best_acc = avg_acc
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc

                }
                torch.save(checkpoint, os.path.join(args.trained_models, "best.pt"))

            checkpoint = {
                "model" : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch" : epoch,
                "best_acc" : best_acc

            }
            torch.save(checkpoint,os.path.join(args.trained_models,"last.pt"))




if __name__ == '__main__':
    args = get_args()
    train(args)