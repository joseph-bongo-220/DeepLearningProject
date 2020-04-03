import numpy as np
import tensorflow as tf
from math import ceil
from sklearn.model_selection import train_test_split
import net
from datetime import datetime
import random

tf.compat.v1.disable_eager_execution()

print('python3 -m tensorboard.main --logdir=./logs/')

def train(inFile):
    print('training...')

    images = np.load(inFile + '_images.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2)

    training = [X_train, y_train]
    test = [X_test, y_test]

    train_net(training, test)

def train_net(training, test, size=512, epochs=100, batch_size=10):

    run_name = datetime.now().strftime(r'%Y-%m-%d_%H:%M')

    X_train, y_train = training
    X_test, y_test = test
    print(X_train.shape)

    border = (X_test.shape[1] - size) // 2
    X_test = X_test[:, border:border+size, border:border+size]

    epoch_size = int(ceil(X_train.shape[0] / batch_size))

    training_set = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=X_train.shape[0])
#            .map(lambda im, lab: tf.py_function(augment, [im, lab, size], [im.dtype, lab.dtype]), num_parallel_calls=4)
            .batch(batch_size)
        .prefetch(1)
    )

    #next_training = training_set.make_one_shot_iterator().get_next()
    next_training = tf.compat.v1.data.make_one_shot_iterator(training_set).get_next()

    inp_var, labels_var, output = net.generate_network(size)
    error_fn, train_fn, metrics = net.generate_functions(inp_var,
                                                         labels_var,
                                                         output)

    config = tf.compat.v1.ConfigProto()
    print('1') 
    with tf.compat.v1.Session(config=config) as sess:
        try:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(epochs):

                sess.run(tf.compat.v1.local_variables_initializer())

                accuracy_fn, accuracy_update = metrics['accuracy']
                auc_fn, auc_update = metrics['AUC']
                print('accuracy_fn')
                for batch in range(epoch_size):
                    batch_images, batch_labels = sess.run(next_training)

                    sess.run([train_fn, accuracy_update, auc_update], {
                    'input:0':batch_images,
                    'labels:0':batch_labels,
                    })

                accuracy = sess.run(accuracy_fn)
                auc = sess.run(auc_fn)

                sess.run(tf.compat.v1.local_variables_initializer())

                for ti, (img, lab) in enumerate(zip(X_test, y_test)):
                    sess.run([accuracy_update, auc_update], {
                    'input:0':img.reshape(1,size,size,-1),
                    'labels:0':[lab],
                    })
                test_accuracy = sess.run(accuracy_fn)
                test_auc = sess.run(auc_fn)

                print(
                    'Epoch {:>3} | Acc: {:>5.3f} (Test: {:>5.3f}) | AUC: {:>5.3f} (Test: {:>5.3f})'
                    .format(epoch, accuracy, test_accuracy, auc, test_auc)
            )
        except tf.errors.OutOfRangeError:
            print('oops')
            pass


def augment(image, labe, size):

    max_displacement = image.shape[0] - size
    displacement_x = int(random.random() * max_displacement)
    displacement_y = int(random.random() * max_displacement)

    image = image[displacement_y:displacement_y+size,
                  displacement_x:displacement_x+size]

    return image, label

train('test')
