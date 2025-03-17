from model import SimpleCNN
from dataset import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor
import torch
import argparse
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Cifar classification")
    parser.add_argument("--batch-size","-b",type=int,default=8)
    parser.add_argument("--epochs","-e",type=int,default=100)
    parser.add_argument("--logging","-l",type=str,default="tensorboard")
    parser.add_argument("--trained-model","-t",type=str,default="trained_model")
    parser.add_argument("--checkpoint","-c",type=str,default=None)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    os.makedirs(args.logging,exist_ok=True)
    os.makedirs(args.trained_model,exist_ok=True)
    writer = SummaryWriter(args.logging)

    train_transform = Compose([
        ToTensor()
    ])
    test_transform = Compose([
        ToTensor()
    ])
    train_dataset = CIFAR10("data",train=True,transform=train_transform)
    test_dataset = CIFAR10("data",train=False,transform=test_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
        drop_last=False
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        best_acc = 0
        start_epoch = 0

    num_iter = len(train_dataloader)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader,colour="green")
        for iter,(img,label) in enumerate(progress_bar):
            img,label = img.to(device),label.to(device)
            predict = model(img)
            loss_value = criterion(predict,label)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1,args.epochs,iter+1,num_iter,loss_value))
            writer.add_scalar("Train/loss",loss_value,epoch * num_iter + iter)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_labels = []
        all_predictions = []
        for img,label in test_dataloader:
            all_labels.extend(label)
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                predict = model(img)
                loss_value = criterion(predict, label)
                indies = torch.argmax(predict.cpu(),dim=1)
                all_predictions.extend(indies)

        all_predictions = [prediction.item() for prediction in all_predictions]
        all_labels = [label.item() for label in all_labels]
        accuracy = accuracy_score(all_labels,all_predictions)
        writer.add_scalar("Val/Accuracy",accuracy,epoch+1)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict
        }
        torch.save(checkpoint,"{}/last_cifar10.pt".format(args.trained_model))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint,"{}/best_cifar10.pt".format(args.trained_model))
            best_acc = accuracy
        print("accuracy {}".format(accuracy))

if __name__ == '__main__':
    args = get_args()
    train(args)