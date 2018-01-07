import selectivesearch
import skimage.io as io
import os
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

cwd = os.getcwd()


def intersection(a_min_x, a_min_y, a_max_x, a_max_y,
                 b_min_x, b_min_y, b_max_x, b_max_y):
    # 输入是两个方框
    if (a_min_x <= b_min_x <= a_max_x or
        a_max_x <= b_max_x <= a_max_x) and (a_min_y <= b_max_y <= a_max_y or
                                            a_min_y <= b_min_y <= a_max_y):
        pass
    else:
        return 0  # 方框不重合

    x_list = sorted([a_min_x, a_max_x, b_min_x, b_max_x])
    y_list = sorted([a_min_y, a_max_y, b_min_y, b_max_y])
    w = x_list[2] - x_list[1]
    h = y_list[2] - y_list[1]  # 中间2个点组成重叠区域
    Si = w * h
    Sa = (a_max_x - a_min_x) * (b_max_y - b_min_y)
    Sb = (b_max_x - b_min_x) * (b_max_y - b_min_y)
    IoU = Si / (Sa + Sb - Si)

    return IoU


def img_resize(img, width=224, height=224, resize_mode=Image.ANTIALIAS,
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


def transfer_to_numpy(img):
    """
    Transfer image to numpy
    :param img: input image
    :return: numpy array
    """
    img.load()
    return np.asarray(img, dtype=np.float32)


def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]


def region_proposals(datafile, num_classes, threshold,
                     svm=False, save=False, save_path='region_proposal.pkl'):
    train_list = open(datafile, 'r')
    labels = []
    images = []
    for line in train_list:
        path, label, ground_truth_rect = line.strip().split(' ')
        img = io.imread(cwd + '/' + path)
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        candidate = set()
        for r in regions:
            if r['rect'] in candidate:
                continue
            if r['size'] < 200:
                continue
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            if len(proposal_img) == 0:
                continue
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            a, b, c = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            image = Image.fromarray(proposal_img)
            resize_proposal = img_resize(image, 96, 96)
            candidate.add(r['rect'])  # 添加候选框
            image = transfer_to_numpy(resize_proposal)
            images.append(image)
            g_rect = ground_truth_rect.split(',')
            g_rect = [int(i) for i in g_rect]
            iou_val = intersection(
                g_rect[0], g_rect[1], g_rect[0] + g_rect[2], g_rect[1] + g_rect[3],
                proposal_vertice[0], proposal_vertice[1], proposal_vertice[2], proposal_vertice[3]
            )
            if iou_val < threshold:
                labels.append(0)
            else:
                labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_dataset_from_pkl(dataset_path):
    images, labels = pickle.load(open(dataset_path, 'rb'))
    return images, labels



