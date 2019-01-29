# -*- coding: utf-8 -*-
"""
This file integrates all functions related to data process.
"""
import os
from tqdm import tqdm
import tensorflow as tf
import sys
import numpy as np
import pickle

# path declaration

# raw data files
tf.flags.DEFINE_string("train_raw", 'data/rawdata/train.csv', 'raw data for training')
tf.flags.DEFINE_string('val_raw', 'data/rawdata/val.csv', 'raw data for validation')
tf.flags.DEFINE_string('val_raw2016', 'data/rawdata/val_2016.csv', 'raw data2 for validation')
tf.flags.DEFINE_string('test_raw', 'data/rawdata/test.csv', 'raw data for test')

# embedding files
tf.flags.DEFINE_string('embedding_src1', 'F:/Data/Glove/glove.6B.300d.txt', '')
tf.flags.DEFINE_string('train_embedding1', 'data/embedding/Glove/train_index1', 'index for the train text(Glove)')
tf.flags.DEFINE_string('val_embedding1', 'data/embedding/Glove/val_index1', 'index for the val text(Glove)')
tf.flags.DEFINE_string('val_embedding2016', 'data/embedding/Glove/val_index2016', 'index for the val text2016(Glove)')
tf.flags.DEFINE_string('test_embedding1', 'data/embedding/Glove/test_index1', 'index for the test text(Glove)')
tf.flags.DEFINE_string('val_labels', 'data/embedding/val_labels', 'labels of the val text')
tf.flags.DEFINE_string('val_embedding2', 'data/embedding/skip-thoughts/val_embd.npy', 'embedding of the val text')
tf.flags.DEFINE_string('test_embedding2', 'data/embedding/skip-thoughts/test_embd.npy', 'embedding of the test text')
tf.flags.DEFINE_string('val_part1', 'data/embedding/Glove/val_part1.npy', 'hold out for train')
tf.flags.DEFINE_string('val_part2', 'data/embedding/Glove/val_part2.npy', 'hold out for val')
tf.flags.DEFINE_integer('train_num', 1200, 'hold out for train')
tf.flags.DEFINE_string('acc_file', 'data/result/acc.txt', 'accuracy')
tf.flags.DEFINE_string('prediction', 'data/result/prediction.txt', 'label result')
tf.flags.DEFINE_string('save_file', 'res/savedModel', 'Save model.')
tf.flags.DEFINE_string('val_st_embedding', 'data/embedding/skip-thoughts/val_preproc.npy', 'skip-thoughts vector for val text')
tf.flags.DEFINE_string('test_st_embedding', 'data/embedding/skip-thoughts/test_preproc.npy', 'skip-thoughts vector for test text')
# training parameters
tf.flags.DEFINE_integer("k", 5, "K most similarity knowledge (default: 5).")
tf.flags.DEFINE_integer("rnn_size", 512, "Neurons number of hidden layer in LSTM cell (default: 100).")
tf.flags.DEFINE_float("margin", 0.2, "Constant of max-margin loss (default: 0.1).")
tf.flags.DEFINE_integer("max_grad_norm", 5, "Control gradient expansion (default: 5).")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 50).")
tf.flags.DEFINE_integer("max_sentence_len", 30, "Maximum number of words in a sentence (default: 100).")
tf.flags.DEFINE_float("dropout_keep_prob", 0.70, "Dropout keep probability (default: 0.5).")
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate (default: 0.4).")
tf.flags.DEFINE_float("lr_down_rate", 0.6, "Learning rate down rate(default: 0.5).")
tf.flags.DEFINE_integer("lr_down_times", 5, "Learning rate down times (default: 4)")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 32, "Evaluate model on dev set after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS  # init flags
FLAGS(sys.argv)  # parse flags

def get_curr_dir():
    curr_dir = os.path.dirname(__file__)
    return curr_dir


def load_embedding():
    """
    :return: the list of word embeddings
    """
    embedding = np.load('data/embedding/Glove/embeddings.npy')
    return embedding


def hold_out():
    """
    split the validation set into 2 parts, one for training and the other part for validation.
    save the result.
    """
    file = open(FLAGS.val_embedding1, 'rb')
    indexs = pickle.load(file)
    total_num = len(indexs)
    train_num = FLAGS.train_num
    val_num = total_num - train_num
    train_indexs = indexs[:train_num]
    val_indexes = indexs[train_num:]
    np.save('data/embedding/Glove/val_part1.npy', np.array(train_indexs))
    np.save('data/embedding/Glove/val_part2.npy', np.array(val_indexes))
    file.close()

    val_st_embd = np.load(FLAGS.val_st_embedding)
    total_num = len(val_st_embd)
    train_num = FLAGS.train_num
    val_num = total_num - train_num
    train_indexs = val_st_embd[:train_num]
    val_indexs = val_st_embd[train_num:]
    np.save('data/embedding/skip-thoughts/val_part1.npy', np.array(train_indexs))
    np.save('data/embedding/skip-thoughts/val_part2.npy', np.array(val_indexs))

def load_data():
    """
    just for Glove method.
    just get the last sentence of the input sentences as the input story.
    load the index list of the embedding matrix, pad, and transform data into the form that we can feed the RCNN.
    :return: np arrays
    """
    max_len = FLAGS.max_sentence_len
    train_indexs = np.load(FLAGS.val_part1)
    val_indexs = np.load(FLAGS.val_part2)
    train_last_sentcs = []
    train_true_ans = []
    train_false_ans = []
    val_last_sentcs = []
    val_true_ans = []
    val_false_ans = []

    labelfile = open(FLAGS.val_labels, 'rb')
    labels = pickle.load(labelfile)
    for (num, line) in enumerate(train_indexs):
        pad1, pad2, pad3 = [40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[3])):
            pad1[i] = line[3][i]
        for i in range(len(line[4])):
            pad2[i] = line[4][i]
        for i in range(len(line[5])):
            pad3[i] = line[5][i]

        train_last_sentcs.append(pad1)
        if labels[num] == '1':
            train_true_ans.append(pad2)
            train_false_ans.append(pad3)
        else:
            train_true_ans.append(pad3)
            train_false_ans.append(pad2)
    for (num, line) in enumerate(val_indexs):
        pad1, pad2, pad3 = [40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[3])):
            pad1[i] = line[3][i]
        for i in range(len(line[4])):
            pad2[i] = line[4][i]
        for i in range(len(line[5])):
            pad3[i] = line[5][i]

        val_last_sentcs.append(pad1)
        if labels[num + FLAGS.trian_num] == '1':
            val_true_ans.append(pad2)
            val_false_ans.append(pad3)
        else:
            val_true_ans.append(pad3)
            val_false_ans.append(pad2)
    labelfile.close()

    test_last_sentcs = []
    test_ans1 = []
    test_ans2 = []
    test_file = open(FLAGS.test_embedding1, 'rb')
    test_indexs = pickle.load(test_file)
    for (num, line) in enumerate(test_indexs):
        pad1, pad2, pad3 = [40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[3])):
            pad1[i] = line[3][i]
        for i in range(len(line[4])):
            pad2[i] = line[4][i]
        for i in range(len(line[5])):
            pad3[i] = line[5][i]

        test_last_sentcs.append(pad1)
        test_ans1.append(pad2)
        test_ans2.append(pad3)

    return np.array(train_last_sentcs), np.array(train_true_ans), np.array(train_false_ans), \
           np.array(val_last_sentcs), np.array(val_true_ans), np.array(val_false_ans), \
           np.array(test_last_sentcs), np.array(test_ans1), np.array(test_ans2)


def load_data_fullstory():
    """
    just for Glove
    get all of the 4 sentences as input story
    :return: np arrays
    """
    max_len = FLAGS.max_sentence_len
    train_indexs = np.load(FLAGS.val_part1)
    val_indexs = np.load(FLAGS.val_part2)
    train_sentcs = []
    train_true_ans = []
    train_false_ans = []
    val_sentcs = []
    val_true_ans = []
    val_false_ans = []

    labelfile = open(FLAGS.val_labels, 'rb')
    labels = pickle.load(labelfile)
    for (num, line) in enumerate(train_indexs):
        pad1, pad2, pad3, pad4, pad5, pad6 = [40000] * max_len, [40000] * max_len, [40000] * max_len, [40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[0])):
            pad1[i] = line[0][i]
        for i in range(len(line[1])):
            pad2[i] = line[1][i]
        for i in range(len(line[2])):
            pad3[i] = line[2][i]
        for i in range(len(line[3])):
            pad4[i] = line[3][i]
        for i in range(len(line[4])):
            pad5[i] = line[4][i]
        for i in range(len(line[5])):
            pad6[i] = line[5][i]

        train_sentcs.append([pad1, pad2, pad3, pad4])
        if labels[num] == '1':
            train_true_ans.append(pad5)
            train_false_ans.append(pad6)
        else:
            train_true_ans.append(pad6)
            train_false_ans.append(pad5)
    for (num, line) in enumerate(val_indexs):
        pad1, pad2, pad3, pad4, pad5, pad6 = [40000] * max_len, [40000] * max_len, [40000] * max_len, [
            40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[0])):
            pad1[i] = line[0][i]
        for i in range(len(line[1])):
            pad2[i] = line[1][i]
        for i in range(len(line[2])):
            pad3[i] = line[2][i]
        for i in range(len(line[3])):
            pad4[i] = line[3][i]
        for i in range(len(line[4])):
            pad5[i] = line[4][i]
        for i in range(len(line[5])):
            pad6[i] = line[5][i]

        val_sentcs.append([pad1, pad2, pad3, pad4])
        if labels[num + 900] == '1':
            val_true_ans.append(pad5)
            val_false_ans.append(pad6)
        else:
            val_true_ans.append(pad6)
            val_false_ans.append(pad5)
    labelfile.close()

    test_sentcs = []
    test_ans1 = []
    test_ans2 = []
    test_file = open(FLAGS.test_embedding1, 'rb')
    test_indexs = pickle.load(test_file)
    for (num, line) in enumerate(test_indexs):
        pad1, pad2, pad3, pad4, pad5, pad6 = [40000] * max_len, [40000] * max_len, [40000] * max_len, [
            40000] * max_len, [40000] * max_len, [40000] * max_len
        for i in range(len(line[0])):
            pad1[i] = line[0][i]
        for i in range(len(line[1])):
            pad2[i] = line[1][i]
        for i in range(len(line[2])):
            pad3[i] = line[2][i]
        for i in range(len(line[3])):
            pad4[i] = line[3][i]
        for i in range(len(line[4])):
            pad5[i] = line[4][i]
        for i in range(len(line[5])):
            pad6[i] = line[5][i]
        test_sentcs.append([pad1, pad2, pad3, pad4])
        test_ans1.append(pad5)
        test_ans2.append(pad6)

    return np.array(train_sentcs), np.array(train_true_ans), np.array(train_false_ans), \
           np.array(val_sentcs), np.array(val_true_ans), np.array(val_false_ans), \
           np.array(test_sentcs), np.array(test_ans1), np.array(test_ans2)


def load_skipthoughts():
    """
    just for skip thoughts method.
    just get the last sentence of the input sentences as the input story.
    :return: np arrays(they are already embeddings)
    """
    labelfile = open(FLAGS.val_labels, 'rb')
    labels = pickle.load(labelfile)
    labelfile.close()
    train_embedding = np.load('data/embedding/skip-thoughts/val_part1.npy')

    train_last_sentc = []
    train_true_ans = []
    train_false_ans = []
    for (num, line) in enumerate(train_embedding):
        train_last_sentc.append(line[3])
        if labels[num] == '1':
            train_true_ans.append(line[4])
            train_false_ans.append(line[5])
        else:
            train_true_ans.append(line[5])
            train_false_ans.append(line[4])

    val_embedding = np.load('data/embedding/skip-thoughts/val_part2.npy')

    val_last_sentc = []
    val_true_ans = []
    val_false_ans = []
    for (num, line) in enumerate(val_embedding):
        val_last_sentc.append(line[3])
        if labels[num + FLAGS.train_num] == '1':
            val_true_ans.append(line[4])
            val_false_ans.append(line[5])
        else:
            val_true_ans.append(line[5])
            val_false_ans.append(line[4])

    test_embedding = np.load('data/embedding/skip-thoughts/test_preproc.npy')

    test_last_sentc = []
    test_ans1 = []
    test_ans2 = []
    for (num, line) in enumerate(test_embedding):
        test_last_sentc.append(line[3])
        test_ans1.append(line[4])
        test_ans2.append(line[5])

    return np.array(train_last_sentc), np.array(train_true_ans), np.array(train_false_ans), \
            np.array(val_last_sentc), np.array(val_true_ans), np.array(val_false_ans), \
            np.array(test_last_sentc), np.array(test_ans1), np.array(test_ans2)



if __name__ == '__main__':
    # hold_out()
    # train_story, train_true_ans, train_false_ans, val_story, val_true_ans, val_false_ans = load_data()
    # print(train_true_ans)
    load_skipthoughts()