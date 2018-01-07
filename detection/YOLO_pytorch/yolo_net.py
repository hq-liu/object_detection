import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from YOLO import config as cfg
import numpy as np


class YOLO():
    def __init__(self, is_training=False, use_gpu=True):
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size ** 2) * (self.num_classes + self.boxes_per_cell * 5)
        # 5表示每个box有5个预测值，xywh和confidence
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_classes
        self.boundary2 = self.boundary1 + self.cell_size ** 2 * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.no_object_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE  # loss 的系数

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

        self.model = models.resnet18(pretrained=True).type(self.FloatTensor)
        self.model.fc = nn.Linear(512, self.output_size).type(self.FloatTensor)
        for param in self.model.parameters():
            param.requires_grad = True

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell
        ), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

    def build_my_cnn(self, output_size):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),

        )

    def get_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 5-D tensor [batch_size, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [batch_size, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [batch_size, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        x_center1, y_center1, w1, h1 = boxes1[:, :, :, :, 0], boxes1[:, :, :, :, 1], \
                                       boxes1[:, :, :, :, 2], boxes1[:, :, :, :, 3]
        x_center2, y_center2, w2, h2 = boxes2[:, :, :, :, 0], boxes2[:, :, :, :, 1], \
                                       boxes2[:, :, :, :, 2], boxes2[:, :, :, :, 3]
        xmin_1, ymin_1, xmax_1, ymax_1 = x_center1 - w1 / 2.0, y_center1 - h1 / 2.0, \
                                         x_center1 + w1 / 2.0, y_center1 + h1 / 2.0
        xmin_2, ymin_2, xmax_2, ymax_2 = x_center2 - w2 / 2.0, y_center2 - h2 / 2.0, \
                                         x_center2 + w2 / 2.0, y_center2 + h2 / 2.0
        l_x = torch.max(xmin_1, xmin_2)
        l_y = torch.max(ymin_1, ymin_2)
        r_x = torch.min(xmax_1, xmax_2)
        r_y = torch.min(ymax_1, ymax_2)

        intersection = (r_y - l_y) * (r_x - l_x)
        S1 = w1 * h1
        S2 = w2 * h2
        union_square = S1 + S2 - intersection
        IoU = intersection / union_square
        IoU = torch.clamp(IoU, min=0.0, max=1.0)
        return IoU

    def get_loss(self, outputs, labels):
        """
        计算loss，分4部分
        :param outputs:网络的输出 shape=(batch_size, cell_size**2 * (num_class + box_per_cell*5)) type=Variable
        :param labels: 真实值 shape=(batch_size, cell_size, cell_size, 5+num_class) type=numpy
        :return:loss
        """
        # labels = torch.from_numpy(labels)

        output_classes = outputs.data[:, :self.boundary1].resize_(
                            [self.batch_size, self.cell_size, self.cell_size, self.num_classes])
        # shape=(batch_size, cell_size, cell_size, num_classes) type=Tensor
        output_confidences = outputs.data[:, self.boundary1:self.boundary2].resize_(
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell]
        )
        # shape=(batch_size, cell_size, cell_size, box_per_cell) type=Tensor
        output_boxes = outputs.data[:, self.boundary2:].resize_(
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4]
        )
        # shape = (batch_size, cell_size, cell_size, box_per_cell, 4) type=Tensor
        # 把输出拆分为3部分，cell的类别，box的confidence和xywh

        ground_truth_response = labels[:, :, :, 0].data.resize_([
            self.batch_size, self.cell_size, self.cell_size, 1
        ])
        # shape=(batch_size, cell_size, cell_size, 1)  估计是cell中每个box共享 type=Tensor
        ground_truth_box = labels[:, :, :, 1:5].data.resize_([
            self.batch_size, self.cell_size, self.cell_size, 1, 4
        ])
        ground_truth_box = torch.cat([ground_truth_box for _ in range(self.boxes_per_cell)], dim=3)
        # shape=(batch_size, cell_size, cell_size, box_per_cell, 4) type=Tensor
        ground_truth_class = labels[:, :, :, 5:].data
        # shape=(batch_size, cell_size, cell_size, num_classes(20)) type=Tensor

        class_delta = ground_truth_response * (ground_truth_class - output_classes)
        class_loss = torch.mean(class_delta ** 2) * self.class_scale


        # offset = torch.from_numpy(self.offset)
        # offset = offset.resize_([1, self.cell_size, self.cell_size, self.boxes_per_cell])
        # offset = torch.cat([offset for _ in range(self.batch_size)], dim=0)
        iou_truth = self.get_iou(output_boxes, ground_truth_box)  #
        object_max = torch.max(iou_truth, dim=3, keepdim=True)[0]  # 取出iou最大的bounding_box 的iou
        object_mask = (iou_truth == object_max)
        object_mask = object_mask.type(self.FloatTensor)
        no_object_mask = torch.ones_like(object_mask) - object_mask

        object_loss = object_mask * (output_confidences - iou_truth)
        object_loss = torch.mean(object_loss ** 2) * self.object_scale

        no_object_loss = no_object_mask * output_confidences
        no_object_loss = torch.mean(no_object_loss ** 2) * self.no_object_scale

        x_hat, y_hat, w_hat, h_hat = output_boxes[:, :, :, :, 0], output_boxes[:, :, :, :, 1], \
                                     output_boxes[:, :, :, :, 2], output_boxes[:, :, :, :, 3]
        w_hat = torch.clamp(w_hat, min=0, max=1e10)
        h_hat = torch.clamp(h_hat, min=0, max=1e10)
        x, y, w, h = ground_truth_box[:, :, :, :, 0], ground_truth_box[:, :, :, :, 1], \
                     ground_truth_box[:, :, :, :, 2], ground_truth_box[:, :, :, :, 3]
        crood_loss = torch.mean(((x_hat - x) ** 2 + (y_hat - y) ** 2) * object_mask) * self.coord_scale
        crood_loss += torch.mean((
                            (torch.sqrt(w_hat) - torch.sqrt(w)) ** 2 + (torch.sqrt(h_hat) - torch.sqrt(h)) ** 2) * \
                           object_mask) * self.coord_scale

        loss = class_loss + object_loss + no_object_loss + crood_loss
        return loss





