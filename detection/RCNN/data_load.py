import os
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf


class Dataloader():
    def __init__(self, datafile, num_classes, img_width, img_height,dataset_save_path=None):
        self.datafile = datafile
        self.num_classes = num_classes
        self.data_save_path = dataset_save_path
        self.img_width = img_width
        self.img_height = img_height

    def load_image(self, image_path):
        """
        Load images with Image
        :param image_path: path
        :return: image
        """
        img = Image.open(image_path)
        return img

    def img_resize(self, img, width=224, height=224, resize_mode=Image.ANTIALIAS,
                   save_img=None):
        """
        Resize image to 224*224
        :param img: input image
        :param width: new image's width
        :param height: new image's height
        :param resize_mode: resize mode
        :param save_img: save new image or not
        :return: new image
        """
        new_img = img.resize((width, height), resize_mode)
        if save_img:
            new_img.save(save_img)
        return new_img

    def transfer_to_numpy(self, img):
        """
        Transfer image to numpy
        :param img: input image
        :return: numpy array
        """
        img.load()
        return np.asarray(img, dtype=np.float32)

    def data_loader(self, save=False):
        """
        Load data from txt
        :param save: save data or not
        :return: list of images and labels
        """
        train_list = open(self.datafile, 'r')
        labels = []
        images = []
        for line in train_list:
            path, label = line.strip().split(' ')
            image = self.load_image(path)
            image = self.img_resize(image, self.img_width, self.img_height)
            image = self.transfer_to_numpy(image)
            images.append(image)
            labels.append(label)

        if save:
            pickle.dump((images,labels), open(self.data_save_path, 'wb'))
        return images, labels

    def load_dataset_from_pkl(self, dataset_path):
        images, labels = pickle.load(open(dataset_path, 'rb'))
        return images, labels

    def get_batch(self, images, labels, batch_size, capacity):
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int64)
        input_queue = tf.train.slice_input_producer([images, labels])
        images = input_queue[0]
        labels = input_queue[1]

        image_batch, label_batch = tf.train.shuffle_batch(
            [images, labels], batch_size=batch_size,
            num_threads=64, capacity=capacity, min_after_dequeue=capacity-1
        )
        label_batch = tf.reshape(label_batch, [batch_size])
        image_batch = tf.cast(image_batch, tf.float32)
        return image_batch, label_batch


