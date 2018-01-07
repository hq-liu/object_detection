import tensorflow as tf
import numpy as np
import YOLO.config as cfg


class YOLO_net():
    def __init__(self, is_training=False):
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

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell
        ), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32,
                                    shape=[None, self.image_size, self.image_size, 3], name='images')
        self.logits = self._build_net(self.images, self.output_size, 0.5)

        if is_training:
            self.labels = tf.placeholder(tf.float32,
                                         shape=[None, self.cell_size, self.cell_size, self.num_classes+5])
            self.get_loss(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)


    def _build_net(self, images, num_outputs, keep_prob):
        with tf.variable_scope('yolo_net'):
            # (none, 448, 448, 3)
            conv1 = tf.layers.conv2d(inputs=images, filters=32, kernel_size=4,
                                     strides=2, activation=tf.nn.relu, name='conv1')
            # (None, 223, 223, 32)
            conv1 = tf.layers.batch_normalization(inputs=conv1, axis=3)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, name='pool1')
            # (None, 111, 111, 32)
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=3,
                                     strides=2, activation=tf.nn.relu, name='conv2')
            # (None, 55, 55, 64)
            conv2 = tf.layers.batch_normalization(inputs=conv2, axis=3)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2, name='pool2')
            # (None, 27, 27, 64)
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=3, strides=2,
                                     activation=tf.nn.relu, name='conv3')
            # (None, 13, 13, 128)
            conv3 = tf.reshape(conv3, shape=[-1, 13 * 13 * 128])
            fc4 = tf.layers.dense(inputs=conv3, units=2000, activation=tf.nn.relu, name='fc4')
            fc4 = tf.layers.dropout(inputs=fc4, rate=keep_prob)
            fc5 = tf.layers.dense(inputs=fc4, units=num_outputs, name='outputs')
            return fc5

    def get_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope('iou'):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def get_loss(self, predicts, labels):
        with tf.variable_scope('loss'):
            # 每个cell的分类
            predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                         shape=[self.batch_size, self.cell_size, self.cell_size, self.num_classes])
            predict_scales = tf.reshape(predicts[:, self.boundary1: self.boundary2],
                                        shape=[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predicts_boxes = tf.reshape(predicts[:, self.boundary2:],
                                        shape=[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], shape=
                                    [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5],
                               shape=[self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, shape=[1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boses_tran = tf.stack([(predicts_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predicts_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predicts_boxes[:, :, :, :, 2]),
                                           tf.square(predicts_boxes[:, :, :, :, 3])])
            predict_boses_tran = tf.transpose(predict_boses_tran, [1, 2, 3, 4, 0])
            iou_predict_truth = self.get_iou(predict_boses_tran, boxes)
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
            no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = no_object_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.no_object_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predicts_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)






