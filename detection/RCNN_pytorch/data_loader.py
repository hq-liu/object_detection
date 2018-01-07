import os
from PIL import Image
import pickle
import numpy as np
from torch.utils.data import Dataset



class Dataloader():
    def __init__(self, datafile, num_classes, img_width, img_height, dataset_save_path=None):
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

    def load_dataset_from_pkl(self):
        images, labels = pickle.load(open(self.data_save_path, 'rb'))
        return images, labels


class MyTrainDataset(Dataset):
    def __init__(self,  transform=None, target_transform=None):
        dataloader = Dataloader('./train_list.txt', 17, 96, 96, './alexnet_dataset.pkl')
        imgs,label = dataloader.load_dataset_from_pkl()
        imgs, label = np.array(imgs), np.array(label)
        self.imgs = imgs
        self.label=label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label=self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


