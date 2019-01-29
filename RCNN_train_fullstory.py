# -*- coding: utf-8 -*-
"""
This file contains the code for training and evaluation.
"""
from data_util import *
from BiLSTM_fullstory import *
import time
import math


def index2embd_story(index, embedding):
    storys_embd = []
    for line in index:
        story_embd = np.zeros(9000, float)
        for sentc in line:
            sentc_embd = embedding[sentc[0]]
            for i in range(1, FLAGS.max_sentence_len):
                sentc_embd = np.concatenate((sentc_embd, embedding[sentc[i]]))
            story_embd += sentc_embd
        storys_embd.append(story_embd / FLAGS.max_sentence_len)
    return np.array(storys_embd)


def index2embd_ans(index, embedding):
    ans_embd = []
    for line in index:
        sentc_embd = embedding[line[0]]
        for i in range(1, FLAGS.max_sentence_len):
            sentc_embd = np.concatenate((sentc_embd, embedding[line[i]]))
        ans_embd.append(sentc_embd)
    return np.array(ans_embd)

def RCNN_train():
    # load pre-trained embedding vector
    print("\n[INFO]: loading embedding...")

    embedding = load_embedding()
    e_len = len(embedding)
    print(e_len)


    # load data
    print("\n[INFO]: loading data...")
    train_story, train_true_ans, train_false_ans, val_story, val_true_ans, val_false_ans,\
        test_story, test_ans1, test_ans2 = load_data_fullstory()

    train_story = index2embd_story(train_story, embedding)
    train_true_ans = index2embd_ans(train_true_ans, embedding)
    train_false_ans = index2embd_ans(train_false_ans, embedding)
    val_story = index2embd_story(val_story, embedding)
    val_true_ans = index2embd_ans(val_true_ans, embedding)
    val_false_ans = index2embd_ans(val_false_ans, embedding)
    test_story = index2embd_story(test_story, embedding)
    test_ans1 = index2embd_ans(test_ans1, embedding)
    test_ans2 = index2embd_ans(test_ans2, embedding)


    print('\n[INFO]: batch info.')
    train_story_len = len(train_story)
    val_story_len = len(val_story)
    test_story_len = len(test_story)
    batch_size = FLAGS.batch_size
    train_batch_num = math.ceil(train_story_len / batch_size)
    val_batch_num = math.ceil(val_story_len / batch_size)
    test_batch_num = math.ceil(test_story_len / batch_size)
    print('train_story_len: %d' % train_story_len)
    print('val_story_len: %d' % val_story_len)
    print('test_story_len: %d' % test_story_len)
    print('train_batch_size: %d' % batch_size)
    print('train_batch_num: %d' % train_batch_num)
    print('validation_batch_num: %d' % val_batch_num)
    print('test_batch_num: %d' % test_batch_num)

    print('[INFO]: batch info ended.')

    def predict():
        print("predicting.")
        scores1 = []
        scores2 = []

        for p_k in tqdm(range(test_batch_num)):
            p_start = p_k * batch_size
            p_end = min((p_k + 1) * batch_size, test_story_len)
            p_story, p_ans1, p_ans2 = [], [], []
            for p_i in range(p_start, p_end):
                p_story.append(test_story[p_i])
                p_ans1.append(test_ans1[p_i])
                p_ans2.append(test_ans2[p_i])
            p_story = np.expand_dims(np.array(p_story), axis=1)
            p_ans1 = np.expand_dims(np.array(p_ans1), axis=1)
            p_ans2 = np.expand_dims(np.array(p_ans2), axis=1)

            test_feed_dict1 = {
                lstm.inputTestQuestions: p_story,
                lstm.inputTestAnswers: p_ans1,
                lstm.dropout: 1.0
            }
            _, score1 = sess.run([globalStep, lstm.result], test_feed_dict1)
            scores1.extend(score1)

            test_feed_dict2 = {
                lstm.inputTestQuestions: p_story,
                lstm.inputTestAnswers: p_ans2,
                lstm.dropout: 1.0
            }
            _, score2 = sess.run([globalStep, lstm.result], test_feed_dict2)
            scores2.extend(score2)

        # get label result
        result = []
        for p_i in range(test_story_len):
            if scores1[p_i] > scores2[p_i]:
                result.append(1)
            else:
                result.append(2)
        file = open(FLAGS.prediction, 'a+')
        for label in result:
            file.write('{}\n'.format(label))
        file.close()

    # evaluating
    def evaluate():
        print("evaluating..")
        scores_true = []
        scores_false = []

        for val_k in tqdm(range(val_batch_num)):
            val_start = val_k * batch_size
            val_end = min((val_k + 1) * batch_size, val_story_len)
            e_story, e_true_ans, e_false_ans = [], [], []
            for val_i in range(val_start, val_end):
                e_story.append(val_story[val_i])
                e_true_ans.append(val_true_ans[val_i])
                e_false_ans.append(val_false_ans[val_i])
            e_story = np.expand_dims(np.array(e_story), axis=1)
            e_true_ans = np.expand_dims(np.array(e_true_ans), axis=1)
            e_false_ans = np.expand_dims(np.array(e_false_ans), axis=1)

            test_feed_dict1 = {
                lstm.inputTestQuestions: e_story,
                lstm.inputTestAnswers: e_true_ans,
                lstm.dropout: 1.0
            }
            _, score_true = sess.run([globalStep, lstm.result], test_feed_dict1)
            scores_true.extend(score_true)

            test_feed_dict2 = {
                lstm.inputTestQuestions: e_story,
                lstm.inputTestAnswers: e_false_ans,
                lstm.dropout: 1.0
            }
            _, score_false = sess.run([globalStep, lstm.result], test_feed_dict2)
            scores_false.extend(score_false)

        # get the accuracy
        acc_num = 0
        for e_i in range(val_story_len):
            if scores_true[e_i] > scores_false[e_i]:
                acc_num += 1
        acc = acc_num / val_story_len
        print('[acc]:   ', acc)
        acc_file = open(FLAGS.acc_file, 'a+')
        acc_file.write('epoch: {}   acc: {}\n'.format(j, acc))
        acc_file.close()


    # start training
    print("\ntraining...")
    with tf.Graph().as_default(), tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.75
        )
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            log_device_placement=False
        )
        with tf.Session(config=session_conf).as_default() as sess:
            globalStep = tf.Variable(0, name="global_step", trainable=False)
            lstm = BiLSTM(
                FLAGS.batch_size,
                FLAGS.max_sentence_len,
                FLAGS.rnn_size,
                FLAGS.margin
            )
            # define training procedure
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), FLAGS.max_grad_norm)
            saver = tf.train.Saver()

            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            print("Writing to {}\n".format(out_dir))

            # get summary
            tf.summary.scalar("loss", lstm.loss)
            summary_op = tf.summary.merge_all()

            summary_dir = os.path.join(out_dir, timestamp)
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

            # training
            sess.run(tf.global_variables_initializer())
            lr = FLAGS.learning_rate
            for down in range(FLAGS.lr_down_times):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer.apply_gradients(zip(grads, tvars))
                trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
                j = 0
                for epoch in range(FLAGS.num_epochs):
                    print('\n[INFO]: start to train epoch %d' % j)

                    for k in tqdm(range(train_batch_num)):
                        start = k*batch_size
                        end = min((k+1) * batch_size, train_story_len)
                        story, trueAnswer, falseAnswer = [], [], []
                        for i in range(start, end):
                            story.append(train_story[i])
                            trueAnswer.append(train_true_ans[i])
                            falseAnswer.append(train_false_ans[i])
                        story = np.array(story)
                        story = np.expand_dims(story, axis=1)
                        trueAnswer = np.array(trueAnswer)
                        trueAnswer = np.expand_dims(trueAnswer, axis=1)
                        falseAnswer = np.array(falseAnswer)
                        falseAnswer = np.expand_dims(falseAnswer, axis=1)
                        feed_dict = {
                            lstm.inputQuestions: story,
                            lstm.inputTrueAnswers: trueAnswer,
                            lstm.inputFalseAnswers: falseAnswer,
                            lstm.dropout: FLAGS.dropout_keep_prob,
                        }
                        _, step, _, _, loss, summary = \
                            sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op],
                                    feed_dict)

                        if step % FLAGS.evaluate_every == 0:
                            print("step:", step, "loss:", loss)
                            # summary_writer.add_summary(summary, step)

                    evaluate()
                    # saver.save(sess, FLAGS.save_file)
                    j += 1

                lr *= FLAGS.lr_down_rate
            evaluate()
            predict()



if __name__ == '__main__':
    RCNN_train()
