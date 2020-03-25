# coding=utf-8
import os
import tensorflow as tf
import numpy as np
import time
import datetime
from helper import batch_gen_with_point_wise, load, prepare, batch_gen_with_single,load_trec_sst2
import operator
from model_cnn import *
# from model_cnn.CNN_origin import CNN as model
from model_cnn.square_fasttext import square_fasttext as model
# from model_cnn.fasttext import fasttext as model
# from model_cnn.Complex_order import CNN as model
import random
from sklearn.metrics import accuracy_score
import pickle
import config
from functools import wraps



now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)
FLAGS = config.flags.FLAGS
FLAGS.flag_values_dict()
log_dir = 'wiki_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/dev_' + FLAGS.data + timeStamp
para_file = log_dir + '/dev_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'
acc_flod=[]


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def predict(sess, cnn, dev, alphabet, batch_size, q_len):
    scores = []
    for data in batch_gen_with_single(dev, alphabet, batch_size, q_len):
        feed_dict = {
            cnn.question: data[0],
            cnn.q_position: data[1],
            cnn.dropout_keep_prob: 1.0
        }
        score = sess.run(cnn.scores, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(dev)])


@log_time_delta
def dev_point_wise():
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        train,dev,test=load_trec_sst2(FLAGS.data)
    else:
        train, dev = load(FLAGS.data)
    q_max_sent_length = max(
        map(lambda x: len(x), train['question'].str.split()))
    print(q_max_sent_length)
    print(len(train))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length', len(train))
    print ('dev length', len(dev))
    embeddings=1
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        alphabet,embeddings = prepare([train, dev,test], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    else:
        # alphabet,embeddings = prepare([train, dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=False, fresh=True)
        alphabet = prepare([train, dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    print ('alphabet:', len(alphabet))
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto()
            session_conf.allow_soft_placement = FLAGS.allow_soft_placement
            session_conf.log_device_placement = FLAGS.log_device_placement
            session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        now = int(time.time())
        timeArray = time.localtime(now)
        timeStamp1 = time.strftime("%Y%m%d%H%M%S", timeArray)
        timeDay = time.strftime("%Y%m%d", timeArray)
        print (timeStamp1)
        with sess.as_default(), open(precision, "w") as log:
            s='embedding_dim:  '+str(FLAGS.embedding_dim)+'\n'+'dropout_keep_prob:  '+str(FLAGS.dropout_keep_prob)+'\n'+'l2_reg_lambda:  '+str(FLAGS.l2_reg_lambda)+'\n'+'learning_rate:  '+str(FLAGS.learning_rate)+'\n'+'batch_size:  '+str(FLAGS.batch_size)+'\n''trainable:  '+str(FLAGS.trainable)+'\n'+'num_filters:  '+str(FLAGS.num_filters)+'\n''data:  '+str(FLAGS.data)+'\n'
            log.write(str(s) + '\n')
            cnn = model(
                max_input_left=q_max_sent_length,
                vocab_size=len(alphabet),
                embeddings=embeddings,
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed=False,
                trainable=FLAGS.trainable,
                dataset=FLAGS.data,
                extend_feature_dim=FLAGS.extend_feature_dim)
            cnn.build_graph()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            acc_max = 0.0000
            for i in range(FLAGS.num_epochs):
                datas = batch_gen_with_point_wise(
                    train, alphabet, FLAGS.batch_size, q_len=q_max_sent_length)
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.input_y: data[1],
                        cnn.q_position: data[2],
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    start = time.time()
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    
                    print("{}: step {}, loss {:g}, acc {:g}  in {} seconds ".format(time_str, step, loss, accuracy,time.time()-start))
                predicted = predict(sess, cnn, train, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted, 1)
                acc_train= accuracy_score(predicted_label,train['flag'])
                predicted_dev = predict(sess, cnn, dev, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted_dev, 1)
                acc_dev= accuracy_score(predicted_label,dev['flag'])
                if acc_dev> acc_max:
                    tf.train.Saver().save(sess, "model_save/model",write_meta_graph=True)
                    acc_max = acc_dev
                print ("{}:train epoch:acc {}".format(i, acc_train))
                print ("{}:dev epoch:acc {}".format(i, acc_dev))
                line2 = " {}:epoch: acc{}".format(i, acc_dev)
                log.write(line2 + '\n')
                log.flush()
            acc_flod.append(acc_max)
            log.close()

if __name__ == '__main__':
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        for attr, value in sorted(FLAGS.__flags.items()):
            print(("{}={}".format(attr.upper(), value)))
        dev_point_wise()
        ckpt = tf.train.get_checkpoint_state("model_save" + '/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        train,dev,test=load_trec_sst2(FLAGS.data)
        q_max_sent_length = max(map(lambda x: len(x), train['question'].str.split()))
        alphabet,embeddings = prepare([train, test,dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)    
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            scores=[]
            question = graph.get_operation_by_name('input_question').outputs[0]
            q_position = graph.get_operation_by_name('q_position').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            for data in batch_gen_with_single(test, alphabet, FLAGS.batch_size, q_max_sent_length):
                feed_dict = {question.name: data[0],q_position.name: data[1],dropout_keep_prob.name: 1.0}
                score = sess.run("output/scores:0", feed_dict)
                scores.extend(score)
            scores=np.array(scores[:len(test)])
            predicted_label = np.argmax(scores, 1)
            acc_test = acc_train= accuracy_score(predicted_label,test['flag'])
            print ("test epoch:acc {}".format(acc_test))
    else:
        for i in range(1,FLAGS.n_fold+1):
            print("{} cross validation ".format(i))
            for attr, value in sorted(FLAGS.__flags.items()):
                print(("{}={}".format(attr.upper(), value)))
            dev_point_wise()
        print("the average acc {}".format(np.mean(acc_flod)))