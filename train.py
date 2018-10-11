from __future__ import print_function
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import pandas
import os

import tensorflow as tf

K.set_image_dim_ordering('tf')

parser = argparse.ArgumentParser()
parser.add_argument('--kbit', type=int, default=13, help='Bits of numbers to detect')
parser.add_argument('--mask', type=int, default=0, help="Masked Bits (with 0) in numbers")
parser.add_argument('--split_ratio', type=float, default=0.67, help="Ratio of train/test split")
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training [default: 16]')
parser.add_argument('--num_epoch', type=int, default=300, help='Batch size during training [default: 30]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_rate', type=float, default=1e-6, help='Decay rate [default: 1e-6]')
parser.add_argument('--log_dir', type=str, default="", help="The path of training log (saving directory)")

FLAGS = parser.parse_args()
kbit = FLAGS.kbit
mask = FLAGS.mask
split_ratio = FLAGS.split_ratio
batch_size = FLAGS.batch_size
nb_epoch = FLAGS.num_epoch
learning_rate = FLAGS.learning_rate
decay_rate = FLAGS.decay_rate
log_dir = FLAGS.log_dir

if log_dir == "":
    log_dir = "logs/kbit_" + str(kbit)

print ("log_dir: " + log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


os.system('cp %s %s' % ("train.py", log_dir)) # bkp of train procedure

FLAGS.log_dir = log_dir
LOG_FOUT = open(os.path.join(log_dir, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
LOG_FOUT.close()


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
            data: B,... numpy array
            label: B, numpy array
        Return:
            shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def load_data(n = 13):
    bits = list(kbits(n))
    bits_string = ''.join(bits)
    bits_mat = (np.array(map(int, list(bits_string))).reshape(-1, n))
    # X = np.expand_dims(bits_mat, axis=2)
    X = bits_mat
    Y = np.array([0, 1] * 2**(n-1)).reshape(2**n, -1)
    Y = np_utils.to_categorical(Y, 2)

    X, Y, _ = shuffle_data(X, Y)

    return X, Y


def load_inputs(n = 13, split_ratio = 0.67):
    """ Generate all numbers of size n bits.
        Input:
            n: bit size (generated numbers)
            split_ratio: train/test split
        Return:
            (X_train, Y_train), (X_test, Y_test)
    """
    X, Y = load_data(n=n)
    split = int(split_ratio * 2**n)
    X_train, Y_train = X[0:split], Y[0:split]
    X_test, Y_test = X[split:], Y[split:]
    print ("Train:", X_train.shape, Y_train.shape)
    print ("Test:", X_test.shape, Y_test.shape)
    print ("Total:", 2**n)
    print ("Ratio:", split_ratio)
    # print (X_train)
    # print (Y_train)

    return (X_train, Y_train), (X_test, Y_test)


def load_inputs_masked(n = 13, mask = 2):
    X_test, Y_test = load_data(n=n)
    bits = list(kbits(n - mask))
    zeros = mask * "0"
    bits_string = zeros.join(bits)
    bits_string = zeros + bits_string
    bits_mat = (np.array(map(int, list(bits_string))).reshape(-1, n))
    X_train = bits_mat
    Y_train = np.array([0, 1] * 2**(n - mask - 1)).reshape(2** (n - mask), -1)
    Y_train = np_utils.to_categorical(Y_train, 2)

    X_train, Y_train, _ = shuffle_data(X_train, Y_train)
    print ("Train:", X_train.shape, Y_train.shape)
    print ("Test:", X_test.shape, Y_test.shape)
    # print (X_train)
    # print (Y_train)

    return (X_train, Y_train), (X_test, Y_test)


def build_model(n, latent_units, lr=learning_rate, decay=decay_rate):
    """ Build MLPs model
        Input:
            n: bit size (generated numbers)
            latent_units: unit number (hidden layer)
        Return:
            (X_train, Y_train), (X_test, Y_test)
    """
    model = Sequential()

    model.add(Dense(kbit, activation='relu', input_shape=(kbit,)))
    # model.add(Dense(latent_units, activation='relu'))
    model.add(Dense(2, activation=None))

    model.add(Activation('softmax'))
    # adam = Adam(lr=lr)
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    return model


def kbits(n):
    """ Generator of all numbers of size n bits.
        Return:
            a generator objective
    """
    for i in xrange(0, 2**n):
        yield '{:0{n}b}'.format(i, n=n)


def train_keras():
    if mask == 0:
        print ("No masking!")
        (X_train, Y_train), (X_test, Y_test) = load_inputs(kbit, split_ratio)
    else:
        print ("Masking: " + str(mask))
        (X_train, Y_train), (X_test, Y_test) = load_inputs_masked(kbit, mask)
    model = build_model(kbit, 1)
    model.summary()
    filepath = os.path.join(log_dir, "weights.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

    callbacks_list = [checkpoint]
    history_callback = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=False,
                                 validation_data=(X_test, Y_test), callbacks=callbacks_list, verbose=2)

    pandas.DataFrame(history_callback.history).to_csv(os.path.join(log_dir, "history.csv"))
    model.save(os.path.join(log_dir, 'model.h5'))


def train_tf():
    # TODO
    # (X_train, Y_train), (X_test, Y_test) = load_inputs(kbit, split_ratio)

    # y = tf.placeholder(tf.int32, [None])
    # x = tf.placeholder(tf.int32, [None, kbit])

    # w1 = tf.truncated_normal(shape=[kbit, 1], stddev=0.01)
    # b1 = tf.constant(0.0, shape=[1])
    # h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # w2 = tf.truncated_normal(shape=[1, 1], stddev=0.01)
    # b2 = tf.constant(0.0, shape=[1])
    # logits = tf.nn.softmax(tf.matmul(h1, w2) + b2)

    # global_step = tf.Variable(0, name='global_step', trainable=False)


    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
    #                 (logits=logits, labels=y, name='xentropy_per_example')
    # loss = tf.reduce_mean(cross_entropy, name='loss')

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    #                                       momenum=0.9, use_nesterov=True)
    # train_op = optimizer.minimize(loss, global_step=global_step)

    pass


if __name__ == "__main__":
    train_keras()

