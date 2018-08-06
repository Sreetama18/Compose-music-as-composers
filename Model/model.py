import csv
import numpy as np
import os
import tensorflow as tf

def read_lines(filename):

    with open(filename, 'rt') as f:

        reader = csv.reader(f)
        lines = list(reader)

    return lines

def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def pad_data(data):
    max = 0

    for i in range(0, data.shape[0]):
        if max < len(data[i]):
            max = len(data[i])

    print(max)
    for j in range(0, data.shape[0]):
        data[j] = pad_along_axis(data[j], max, 0)

    return data


def batched(data, target, batch_size):
    epoch = 0
    offset = 0
    while True:
        old_offset = offset
        offset = (offset + batch_size) % (target.shape[0] - batch_size)

        # Offset wrapped around to the beginning so new epoch
        if offset < old_offset:
            # New epoch, need to shuffle data
            shuffled_indices = np.random.permutation(len(data))

            data = data[shuffled_indices]
            target = target[shuffled_indices]

            epoch += 1

        batch_data = data[offset:(offset + batch_size)]

        batch_target = target[offset:(offset + batch_size)]

        yield batch_data, batch_target, epoch


def main():

    print('Enter the directory for set1')
    path = input()
    data1 = []
    for root, dirs, files in os.walk(path):
        for song in files:
            data1.append(read_lines(path + song))

    data1 = pad_data(data1)

    one_hot_target1 = np.zeros([data1.shape[0], 3])

    for i in range(0, one_hot_target1.shape[0]):
        one_hot_target1[i] = 1

    print(one_hot_target1.shape)

    print('Enter the directory for set2')
    path = input()
    data2 = []
    for root, dirs, files in os.walk(path):
        for song in files:
            data2.append(read_lines(path + song))

    data2 = pad_data(data2)

    one_hot_target2 = np.zeros([data2.shape[0], 3])

    for i in range(0, one_hot_target2.shape[0]):
        one_hot_target2[i] = 2

    print(one_hot_target2.shape)

    print('Enter the directory for set3')
    path = input()
    data3 = []
    for root, dirs, files in os.walk(path):
        for song in files:
            data3.append(read_lines(path + song))

    data3 = pad_data(data3)

    one_hot_target3 = np.zeros([data3.shape[0], 3])

    for i in range(0, one_hot_target3.shape[0]):
        one_hot_target3[i] = 3

    print(one_hot_target3.shape)

    minLength = min(one_hot_target1.shape[0], one_hot_target2.shape[0], one_hot_target3.shape[0])

    data1 = np.delete(data1, np.s_[minLength:], 0)
    data2 = np.delete(data2, np.s_[minLength:], 0)
    data3 = np.delete(data3, np.s_[minLength:], 0)


    one_hot_target1 = np.delete(one_hot_target1, np.s_[minLength:], 0)
    one_hot_target2 = np.delete(one_hot_target2, np.s_[minLength:], 0)
    one_hot_target3 = np.delete(one_hot_target3, np.s_[minLength:], 0)

    print(data1.shape[0])
    print(data2.shape[0])
    print(data3.shape[0])

    print(one_hot_target1.shape[0])
    print(one_hot_target2.shape[0])
    print(one_hot_target3.shape[0])

    data = np.hstack((data1, data2, data3))
    one_hot_target = np.vstack((one_hot_target1, one_hot_target2, one_hot_target3))

    print("data.shape = ", data.shape)
    print("one_hot_target.shape = ", one_hot_target.shape)


    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data[shuffled_indices]
    shuffled_target = one_hot_target[shuffled_indices]

    split = int(0.66 * len(shuffled_data))
    train_data = shuffled_data[:split]
    test_data = shuffled_data[split:]

    train_target = shuffled_target[:split]
    test_target = shuffled_target[split:]

    print("train_data.shape = ", train_data.shape)
    print("test_data.shape = ", test_data.shape)
    print("train_target.shape = ", train_target.shape)
    print("test_target.shape = ", test_target.shape)

    num_steps = train_data.shape[0]
    num_inputs = train_data[0].shape[0]
    num_classes = 3

    print("num_steps = ", num_steps)
    print("num_inputs = ", num_inputs)

    tf.reset_default_graph()

    X = tf.placeholder(tf.float64, [None, num_steps, num_inputs])

    y = tf.placeholder(tf.float64, [None, num_steps, num_classes])

    # All real characters will have a max value of 1, padded characters will be represented by 0s
    used = tf.sign(tf.reduce_max(tf.abs(X), reduction_indices=2))

    # Sum up the number of real characters for each word
    length = tf.reduce_sum(used, reduction_indices=1)
    sequence_length = tf.cast(length, tf.int32)
    num_neurons = 300
    cell = tf.nn.rnn_cell.GRUCell(num_neurons)
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, sequence_length=sequence_length)
    weight = tf.Variable(tf.truncated_normal([num_neurons, num_classes], stddev=0.01, dtype=tf.float64))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float64))
    flattened_output = tf.reshape(output, [-1, num_neurons])
    logits = tf.matmul(flattened_output, weight) + bias
    logits_reshaped = tf.reshape(logits, [-1, num_steps, num_classes])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    loss = tf.reduce_mean(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(y, 2), tf.argmax(logits_reshaped, 2))
    mistakes = tf.cast(mistakes, tf.float64)
    mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
    mistakes *= mask
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.cast(sequence_length, tf.float64)

    error = tf.reduce_mean(mistakes)

    optimizer = tf.train.RMSPropOptimizer(0.002)

    gradient = optimizer.compute_gradients(loss)

    optimize = optimizer.apply_gradients(gradient)

    batch_size = 20
    batches = batched(train_data, train_target, batch_size)

    epochs = 5

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for index, batch in enumerate(batches):
            batch_data = batch[0]
            batch_target = batch[1]

            epoch = batch[2]

            if epoch >= epochs:
                break

            feed = {X: batch_data, y: batch_target}
            train_error, _ = sess.run([error, optimize], feed)
            print('{}: {:3.6f}%'.format(index + 1, 100 * train_error))

        test_feed = {X: test_data, y: test_target}
        test_error, _ = sess.run([error, optimize], test_feed)

        print('Test error: {:3.6f}%'.format(100 * test_error))


if __name__=="__main__":

    main()