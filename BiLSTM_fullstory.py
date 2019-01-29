# -*- coding: utf-8 -*-
"""
This file contains the structure of LSTM.
"""
import tensorflow as tf


class BiLSTM(object):
    def __init__(self, batch_size, max_sentence_len, rnn_size, margin):
        self.batch_size = batch_size
        self.max_sentence_len = max_sentence_len
        # self.embeddings = embeddings
        self.rnn_size = rnn_size
        self.margin = margin

        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.inputQuestions = tf.placeholder(tf.float32, shape=[None, 1, 9000])
        self.inputTrueAnswers = tf.placeholder(tf.float32, shape=[None, 1, 9000])
        self.inputFalseAnswers = tf.placeholder(tf.float32, shape=[None, 1, 9000])
        self.inputTestQuestions = tf.placeholder(tf.float32, shape=[None, 1, 9000])
        self.inputTestAnswers = tf.placeholder(tf.float32, shape=[None, 1, 9000])

        querys = tf.convert_to_tensor(self.inputQuestions)
        true_answers = tf.convert_to_tensor(self.inputTrueAnswers)
        false_answers = tf.convert_to_tensor(self.inputFalseAnswers)
        test_querys = tf.convert_to_tensor(self.inputTestQuestions)
        test_answers = tf.convert_to_tensor(self.inputTestAnswers)
        # embedding layer
        # with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
        #     tf_embedding = tf.Variable(self.embeddings, dtype=tf.float32, name='embedding_matrix', trainable=True)
        #
        #     querys = tf.nn.embedding_lookup(tf_embedding, train_storys)
        #     true_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTrueAnswers)
        #     false_answers = tf.nn.embedding_lookup(tf_embedding, self.inputFalseAnswers)
        #
        #     test_querys = tf.nn.embedding_lookup(tf_embedding, test_storys)
        #     test_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTestAnswers)
        # LSTM
        with tf.variable_scope("LSTM_scope", reuse=None):

            query_lstm = self.biLSTMCell(querys, self.rnn_size)
            query_cnn = tf.nn.tanh(self.max_pooling(query_lstm))     # [batch_size*max_sentence_len, word embedding size] feature vector
        with tf.variable_scope("LSTM_scope", reuse=True):
            true_answer_lstm = self.biLSTMCell(true_answers, self.rnn_size)
            true_answer_cnn = tf.nn.tanh(self.max_pooling(true_answer_lstm))
            false_answer_lstm = self.biLSTMCell(false_answers, self.rnn_size)
            false_answer_cnn = tf.nn.tanh(self.max_pooling(false_answer_lstm))

            test_query_lstm = self.biLSTMCell(test_querys, self.rnn_size)
            test_query_cnn = tf.nn.tanh(self.max_pooling(test_query_lstm))
            test_answer_lstm = self.biLSTMCell(test_answers, self.rnn_size)
            test_answer_cnn = tf.nn.tanh(self.max_pooling(test_answer_lstm))

        self.trueCosSim = self.get_cosine_similarity(query_cnn, true_answer_cnn)
        self.falseCosSim = self.get_cosine_similarity(query_cnn, false_answer_cnn)
        self.loss = self.get_loss(self.trueCosSim, self.falseCosSim, self.margin)

        self.result = self.get_cosine_similarity(test_query_cnn, test_answer_cnn)

    def biLSTMCell(self, x, hidden_size):
        """
        :param x: input
        :param hidden_size: the hidden_dize of rnn
        :return: lstm result
        """
        input = tf.transpose(x, [1, 0, 2])
        input = tf.unstack(input)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.8, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout,
                                                     output_keep_prob=self.dropout)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.8, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout,
                                                     output_keep_prob=self.dropout)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    def get_cosine_similarity(q, a):    # return a [batch_size*max_sentence_len, 1] vector
        """
        :param q: query
        :param a: answer
        :return: the cosine similarity between the query and answer
        """
        query = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        ans = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cos_sim = tf.div(mul, tf.multiply(query, ans))
        return cos_sim

    @staticmethod
    def max_pooling(lstm_result):
        """
        :param lstm_result: the result of lstm
        :return: pooling result; a [batch_size*max_sentence_len, word embedding size] feature vector
        """
        height = int(lstm_result.get_shape()[1])   # the max_sentence_len of query
        width = int(lstm_result.get_shape()[2])    # the word embedding size
        lstm_result = tf.expand_dims(lstm_result, -1)
        output = tf.nn.max_pool(lstm_result, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def get_loss(trueCosSim, falseCosSim, margin):
        """
        :param trueCosSim: the similarity between the query and the true answer
        :param falseCosSim: the similarity between the query and the false answer
        :param margin: default: 1
        :return: the max-margin loss
        """
        Margin = tf.fill(tf.shape(trueCosSim), margin)
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        with tf.name_scope("loss"):
            # max-margin losses = max(0, margin - true + false)
            losses = tf.maximum(zero, tf.subtract(Margin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss
