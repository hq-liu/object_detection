import tensorflow as tf
import xml.etree.ElementTree as ET

data_dir = '/home/lhq/VOCdevkit/VOC2007'
image_idx = '000001'
image_path = '%s/JPEGImages/%s.jpg'%(data_dir, image_idx)
annotation_path = '%s/Annotations/%s.xml'%(data_dir, image_idx)

image_data = tf.gfile.FastGFile(image_path, 'rb').read()

tree = ET.parse(annotation_path)
root = tree.getroot()
size = root.find('size')
shape = [int(size.find('height').text),
         int(size.find('width').text),
         int(size.find('depth').text)]

bboxes = []
labels = []
labels_text = []
difficult = []
truncated = []
for obj in root.findall('object'):
    label = obj.find('name').text
    print(label)
    labels.append(1)
    labels_text.append(label.encode('ascii'))

    if obj.find('difficult') is not None:
        difficult.append(int(obj.find('difficult').text))
    else:
        difficult.append(0)

    if obj.find('truncated') is not None:
        truncated.append(int(obj.find('truncated').text))
    else:
        truncated.append(0)

    bbox = obj.find('bndbox')
    bboxes.append((float(bbox.find('ymin').text) / shape[0],
                   float(bbox.find('xmin').text) / shape[1],
                   float(bbox.find('ymax').text) / shape[0],
                   float(bbox.find('xmax').text) / shape[1]))


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_example(image_data, labels, labels_text, bboxes, shape,
                       difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax])]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Feature(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/labels': int64_feature(labels),
        'image/object/bbox/labels_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))
    return example


tf_filename = 'voc'
