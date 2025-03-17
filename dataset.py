from torchvision.datasets import CIFAR10
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose,ToTensor,RandomAffine,ColorJitter
import pickle

class CIFAR10(Dataset):
    def __init__(self, root="data", train=True,transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.transform = transform
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.images.extend(dict[b'data'])
                self.labels.extend(dict[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = self.images[idx].reshape((32, 32, 3)).astype(np.float32)  # RGB
        image = np.array(self.images[idx],dtype=np.uint8).reshape((32,32,3))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        # RandomAffine(
        #     degrees=(-1, 1),
        #     translate=(0.05, 0.05),
        #     scale=(0.85, 1.15),
        #     shear=5
        # ),
        # ColorJitter(
        #     brightness=0.5,
        #     contrast=0.5,
        #     saturation=0.25,
        #     hue=0.5
        # )

    ])

    # os.makedirs("data", exist_ok=True)
    # train_dataset = CIFAR10(root="data", train=True, download=False)
    # categories = ["airplane", "auto", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    #
    # img, label = train_dataset.__getitem__(789)
    # img = np.array(img)
    # img = cv2.resize(img, (224, 224))
    # cv2.imshow("a", img)
    # print("label: {} - class {}".format(label, categories[label]))
    # cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
# cv2.destroyAllWindows()