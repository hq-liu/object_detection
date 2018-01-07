from RCNN.preprocess2 import *
import tensorflow as tf
import math


def fine_tune_mycnn(inputs, num_classes):
    with tf.variable_scope('my_cnn_refine'):
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
        fc4 = tf.layers.dropout(inputs=fc4, rate=0.5)
        head1 = tf.layers.dense(inputs=fc4, units=num_classes, name='head1')
        return head1


def refine(num_classes, learning_rate, batch_size):
    saver = tf.train.import_meta_graph('./logs_1/mycnn_net.ckpt.meta')
    images, labels = load_dataset_from_pkl('./region_proposal.pkl')
    images, labels = np.array(images), np.array(labels, dtype=np.int64)
    # X = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
    # y = tf.placeholder(tf.int64, shape=[None])
    # y_ = fine_tune_mycnn(X, 3)

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./logs_1/'))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name('input:0')
    y_ = graph.get_tensor_by_name('label:0')
    convnet = graph.get_tensor_by_name('my_cnn/fc4/Relu:0')
    output = tf.layers.dense(inputs=convnet, units=num_classes, name='head1', kernel_initializer=
                             tf.random_normal_initializer(stddev=0.3, mean=0), bias_initializer=
                             tf.constant_initializer(0.1))
    y = output
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
    loss = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)  # compare predict result and label
    correct_prediction = tf.cast(correct_prediction, tf.float32)  # convert bool to float
    accuracy1 = tf.reduce_mean(correct_prediction)  # calculate mean accuracy

    train_indicies = np.arange(images.shape[0])
    np.random.shuffle(train_indicies)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,name='optim2').minimize(loss)
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("logs_2/", sess.graph)
    for ep in range(10):
        for i in range(int(math.ceil(images.shape[0] / batch_size))):
            start_idx = (i * batch_size) % images.shape[0]
            idx = train_indicies[start_idx: start_idx + batch_size]
            _, acc, loss_ = sess.run([train_op, accuracy1, loss],
                                     feed_dict={X: images[idx, :], y_: labels[idx]})
            print('epoch: ', ep)
            print('accuracy = ', acc)
            print('loss=',loss_)
            print()
    saver.save(sess, './logs_2/refine_net.ckpt')


refine(3, 1e-3, 64)
