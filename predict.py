from model import SimpleCNN
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
def get_args():
    parser = argparse.ArgumentParser(description="Test Cifar10")
    parser.add_argument("--checkpoint","-c",type=str,default="trained_model/best_cifar10.pt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(10).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("Not found checkpoint!")
    categories = ["airplane", "auto", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    image = cv2.imread("cat.jpeg")
    ori_image = image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(32,32))
    image = np.transpose(image,(2,0,1))/255.
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0).to(device).float()

    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)
    indies = torch.argmax(probs,dim=1)
    class_name = categories[indies]
    print("class_name: {}".format(class_name))
    percent = probs[0][indies] * 100
    print(percent.item())
    cv2.imshow("cifar10",ori_image)
    cv2.waitKey(0)


