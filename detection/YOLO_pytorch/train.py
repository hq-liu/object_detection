import torch
from YOLO import config as cfg
from torch import optim
from torch.autograd import Variable
from YOLO_pytorch import yolo_net
from YOLO.pascal_voc import *


class Solver():
    def __init__(self, net, data, use_gpu=True):
        self.net = net
        self.data = data
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER

        self.optimizer = optim.SGD(self.net.model.parameters(), lr=self.initial_learning_rate)

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

    def train(self):
        for ep in range(self.max_iter):
            images, labels = self.data.get()
            images = np.transpose(images, (0, 3, 1, 2))
            images, labels = Variable(torch.from_numpy(images)).type(self.FloatTensor), \
                             Variable(torch.from_numpy(labels)).type(self.FloatTensor)
            outputs = self.net.model(images)
            loss = self.net.get_loss(outputs, labels)
            loss = Variable(torch.FloatTensor([loss]), requires_grad=True).type(self.FloatTensor)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            print('epoch:',ep)
            print('loss:', loss.data.cpu().numpy())
            print('-' * 10)
            torch.save(self.net.model.state_dict(), 'model.pkl')

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    yolo_net = yolo_net.YOLO(use_gpu=use_gpu)
    data = pascal_voc('train')
    solver = Solver(yolo_net, data, use_gpu)
    solver.train()

