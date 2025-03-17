import torch
import torch.nn as nn
import cv2
import numpy as np
class SimpleCNN(nn.Module):
    # 32 x 32
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3,out_channels=8)
        self.conv2 = self.make_block(in_channels=8,out_channels=16)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=8*8*16,out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,out_features=num_classes)
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x




if __name__ == '__main__':
    input_tensor = torch.rand((1, 3, 32, 32))  # random tensor từ 0 đến 1
    img = cv2.imread("/home/amin/PycharmProjects/PythonProject/test.jpg")
    img = cv2.resize(img, (32, 32))

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)

    model = SimpleCNN(num_classes=10)
    output = model(input_tensor)
    print("goc {}".format(output))
    soft_max = nn.Softmax(dim=1)
    out = soft_max(output)
    max_idx_out = torch.argmax(output, dim=1)
    max_idx_soft = torch.argmax(out, dim=1)
    print("soft {}".format(max_idx_soft.item()))
    print("argmax {}".format(max_idx_out))
