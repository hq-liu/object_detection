import tensorflow as tf
from RCNN.data_load import *
import math


def alexnet(input_images, num_classes):
    with tf.variable_scope('alex_net'):
        # (none, 224, 224, 3)
        conv1 = tf.layers.conv2d(inputs=input_images, filters=96, kernel_size=11,
                                 strides=4, activation=tf.nn.relu, padding='same')
        # (none, 55, 55, 96)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2)
        # (none, 27, 27, 96)
        pool1 = tf.nn.lrn(input=pool1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=5, strides=1,
                                 activation=tf.nn. relu, padding='same')
        # (none, 27, 27, 256)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2)
        # (none, 13, 13, 256)
        pool2 = tf.nn.lrn(input=pool2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=3, strides=1,
                                 activation=tf.nn.relu, padding='same')
        # (none, 13, 13, 384)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=3, strides=1,
                                 activation=tf.nn.relu, padding='same')
        # (none, 13, 13, 384)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=3, strides=1,
                                 activation=tf.nn.relu, padding='same')
        # (none, 13, 13, 256)
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2)
        # (none, 6, 6, 256)
        pool5 = tf.reshape(pool5, shape=[-1, 9216])
        # (none, 9216)
        fc6 = tf.layers.dense(inputs=pool5, units=4096, activation=tf.nn.relu)
        fc7 = tf.layers.dense(inputs=fc6, units=4096, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=fc7, units=num_classes, activation=tf.nn.relu)
        output = tf.nn.softmax(output)
        return output


def my_cnn(inputs, num_classes):
    with tf.variable_scope('my_cnn'):
        # (none, 96, 96, 3)
        conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=2, strides=2, name='conv1')
        # (none, 48, 48, 32)
        conv1 = tf.layers.batch_normalization(inputs=conv1, axis=3, name='bn1')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, name='pool1')
        # (none, 24, 24, 32)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=4, strides=2, name='conv2')
        # (none, 11, 11, 64)
        conv2 = tf.layers.batch_normalization(inputs=conv2, axis=3, name='bn2')
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, name='pool2')
        # (none, 5, 5, 64)
        pool2 = tf.reshape(pool2, shape=[-1, 1600])
        fc3 = tf.layers.dense(inputs=pool2, units=512, activation=tf.nn.relu, name='fc3')
        fc3 = tf.layers.dropout(inputs=fc3, rate=0.5)
        fc4 = tf.layers.dense(inputs=fc3, units=512, activation=tf.nn.relu, name='fc4')
    with tf.variable_scope('output'):
        fc5 = tf.layers.dense(inputs=fc4, units=num_classes, name='fc5')
        return fc5


def train_alexnet(batch_size,learning_rate, max_epoch, save_epoch):
    dataloader = Dataloader('./train_list.txt', 17, 224, 224, './alexnet_dataset.pkl')
    images, labels = dataloader.load_dataset_from_pkl('./alexnet_dataset.pkl')
    images, labels = np.array(images), np.array(labels)
    # image_batch, label_batch = dataloader.get_batch(images, labels, batch_size=batch_size, capacity=4000)
    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    y = tf.placeholder(dtype=tf.int64, shape=[None,])
    logits = alexnet(X, 17)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    # loss = -tf.reduce_sum(y * tf.log(logits))

    train_indicies = np.arange(images.shape[0])
    np.random.shuffle(train_indicies)
    train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(max_epoch):
            for i in range(int(math.ceil(images.shape[0]/batch_size))):
                start_idx = (i * batch_size) % images.shape[0]
                idx = train_indicies[start_idx: start_idx+batch_size]
                # batch_indices = np.random.choice(images.shape[0], batch_size)
                # image_batch, label_batch = images[batch_indices], labels[batch_indices]
                _, acc, loss_ = sess.run([train_op, accuracy, loss],
                                         feed_dict={X: images[idx, :], y: labels[idx]})
            print('epoch: ', ep)
            print('accuracy = ', acc)
            print('loss=',loss_)
            print()
            if ep % save_epoch == 0:
                print('epoch: ', ep)
                print('accuarcy = ', acc)
                saver.save(sess, './logs/alex_net.ckpt')
                print('save finished')


def train_mycnn(batch_size, learning_rate, max_epoch, save_epoch):
    dataloader = Dataloader('./train_list.txt', 17, 96, 96, './my_dataset.pkl')
    images, labels = dataloader.load_dataset_from_pkl('./my_dataset.pkl')
    images, labels = np.array(images), np.array(labels)
    # image_batch, label_batch = dataloader.get_batch(images, labels, batch_size, 4000)
    X = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3], name='input')
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
    logits = my_cnn(X, 17)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 17), logits=logits)
    loss = tf.reduce_mean(loss)
    # loss = -tf.reduce_sum(y * tf.log(logits))

    train_indicies = np.arange(images.shape[0])
    np.random.shuffle(train_indicies)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("logs_1/", sess.graph)
        for ep in range(max_epoch):
            for i in range(int(math.ceil(images.shape[0]/batch_size))):
                start_idx = (i * batch_size) % images.shape[0]
                idx = train_indicies[start_idx: start_idx+batch_size]
                # batch_indices = np.random.choice(images.shape[0], batch_size)
                # image_batch, label_batch = images[batch_indices], labels[batch_indices]
                _, acc, loss_ = sess.run([train_op, accuracy, loss],
                                         feed_dict={X: images[idx, :], y: labels[idx]})
            print('epoch: ', ep)
            print('accuracy = ', acc)
            print('loss=',loss_)
            print()
            if ep % save_epoch == 0:
                print('epoch: ', ep)
                print('accuarcy = ', acc)
                saver.save(sess, './logs_1/mycnn_net.ckpt')
                print('save finished')


if __name__ == '__main__':
    train_mycnn(64, 1e-3, 200, 10)

