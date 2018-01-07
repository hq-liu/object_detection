import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from RCNN_pytorch.data_loader import *
import time

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_data = MyTrainDataset(transform=img_transform)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

model = models.alexnet(pretrained=True)
use_gpu = torch.cuda.is_available()


def train():
    dataloader = Dataloader('./train_list.txt', 17, 224, 224, './alexnet_dataset.pkl')

    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(200):
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 1):
            img, label = data
            print(label)
            # img = img.view(img.size(0), -1)  # 将图片展开成 64x64
            img = Variable(img).type(FloatTensor)
            label = Variable(label).type(LongTensor)
            # 向前传播
            out = model(img)
            loss = loss_fn(out, label)
            running_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.data[0]
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 300 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, 200, running_loss / (64 * i),
                    running_acc / (64 * i)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_data)), running_acc / (len(
                train_data))))

train()
