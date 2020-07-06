# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

sys.path.append('./utils')
from tf_funcs import *
from prepare_data import *


FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of sentences per document')
## model struct ##
tf.app.flags.DEFINE_string('model_type', 'Inter-EC', 'model type: Indep, Inter-CE, Inter-EC')
tf.app.flags.DEFINE_string('trans_type', 'cross_road', 'transformer type: cross_road, window_constrained')
tf.app.flags.DEFINE_integer('window_size', 3, 'window_size')
tf.app.flags.DEFINE_integer('trans_iter', 2, 'number of cross-road 2D transformer layers')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
tf.app.flags.DEFINE_string('scope', 'TEMP', 'scope')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'keep prob for word embedding')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'keep prob for softmax layer')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('emo', 1., 'loss weight of emotion ext.')
tf.app.flags.DEFINE_float('cause', 1., 'loss weight of cause ext.')
tf.app.flags.DEFINE_float('pair', 1., 'loss weight of pair ext.')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold for pair ext.')
tf.app.flags.DEFINE_integer('feature_num', 30, 'feature vector length of pairs')
tf.app.flags.DEFINE_integer('training_iter', 20, 'number of training iter')


def build_subtasks(x, sen_len, doc_len, is_training):
    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'):  
            inputs = biLSTM(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
        # inputs shape:        [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s

    def emo_cause_prediction(s_ec, is_training, name):
        s1 = tf.nn.dropout(s_ec, keep_prob = is_training * FLAGS.keep_prob2 + (1.-is_training))
        s1 = tf.reshape(s1, [-1, 2 * FLAGS.n_hidden])
        w_ec = get_weight_varible('softmax_w_'+name, [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_ec = get_weight_varible('softmax_b_'+name, [FLAGS.n_class])
        pred_ec = tf.nn.softmax(tf.matmul(s1, w_ec) + b_ec)
        pred_ec = tf.reshape(pred_ec, [-1, FLAGS.max_doc_len, FLAGS.n_class])
        return pred_ec, w_ec, b_ec

    with tf.name_scope('emotion_prediction'):
        s1 = get_s(x, sen_len, name='word_encode_emo')
        s_emo = biLSTM(s1, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_emo')
        pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

    with tf.name_scope('cause_prediction'):
        s1 = get_s(x, sen_len, name='word_encode_cause')
        feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
        if FLAGS.model_type in ['Inter-CE', 'Inter-EC']:
            s1 = tf.concat([s1, pred_emo], 2) * feature_mask
        s_cause = biLSTM(s1, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_cause')
        pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')

    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
    if FLAGS.model_type in ['Inter-CE']:
        return pred_cause, pred_emo, s_cause, s_emo, reg
    return pred_emo, pred_cause, s_emo, s_cause, reg

def pair_prediction(inputs, feature_num, scope="pair_prediction"):
    inputs = tf.reshape(inputs, [-1, feature_num])
    w_pair = get_weight_varible(scope+'_softmax_w_pair', [feature_num, FLAGS.n_class])
    b_pair = get_weight_varible(scope+'_softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(inputs, w_pair) + b_pair)
    reg_tmp = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
    return pred_pair, reg_tmp

def build_maintask_WC(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding):
    ####################################### pair features ################################################
    print('pair features')
    batch = tf.shape(s_emo)[0]
    conc0 = tf.zeros([batch, 2 * FLAGS.n_hidden])
    pair_x = []
    for i in range(FLAGS.max_doc_len):
        for j in range(i-FLAGS.window_size,i+FLAGS.window_size+1):
            conc_i = s_emo[:,i,:]
            conc_j = s_cause[:,j,:] if j in range(FLAGS.max_doc_len) else conc0
            pred_emo_feature_i = pred_emo_feature[:,i,:]
            pred_cause_feature_j = pred_cause_feature[:,j,:] if j in range(FLAGS.max_doc_len) else conc0[:,:2]
            relative_pos = tf.nn.embedding_lookup(pos_embedding, tf.ones([batch], tf.int32) * (j-i+100) )
            ns = tf.concat([conc_i, conc_j, pred_emo_feature_i, pred_cause_feature_j, relative_pos], 1)
            pair_x.append(ns)
    pair_x = tf.transpose(tf.cast(pair_x, tf.float32), perm=[1, 0, 2])
    pair_x = tf.layers.dense(pair_x, FLAGS.feature_num, use_bias=True, activation=tf.nn.relu)
    print('pair features Done!')

    ########################### pair interaction & prediction ########################################################################
    print('pair interaction')
    for i in range(FLAGS.trans_iter):
        pair_x = standard_trans(pair_x, n_hidden = FLAGS.feature_num, n_head = 1, scope="standard_trans{}".format(i))
    print('pair interaction Done!')

    pred_pair, reg_tmp = pair_prediction(pair_x, FLAGS.feature_num, scope="pair_prediction")
    pred_pair = tf.reshape(pred_pair, [-1, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    return pred_pair, reg_tmp

def build_maintask_CR(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding, doc_len):
    ####################################### pair features ################################################
    print('pair features')
    feature_num = FLAGS.feature_num
    s_emo = tf.layers.dense(s_emo, feature_num, use_bias=True)
    s_cause = tf.layers.dense(s_cause, feature_num, use_bias=True)
    pred_emo_feature = tf.layers.dense(pred_emo_feature, feature_num, use_bias=True)
    pred_cause_feature = tf.layers.dense(pred_cause_feature, feature_num, use_bias=True)
    pos_embedding = tf.layers.dense(pos_embedding, feature_num, use_bias=True)
    ## 
    s_emo = tf.tile(tf.reshape(s_emo, [-1, FLAGS.max_doc_len, 1, feature_num]), [1,1,FLAGS.max_doc_len,1])
    s_cause = tf.tile(tf.reshape(s_cause, [-1, 1, FLAGS.max_doc_len, feature_num]), [1,FLAGS.max_doc_len,1,1])
    pred_emo_feature = tf.tile(tf.reshape(pred_emo_feature, [-1, FLAGS.max_doc_len, 1, feature_num]), [1,1,FLAGS.max_doc_len,1])
    pred_cause_feature = tf.tile(tf.reshape(pred_cause_feature, [-1, 1, FLAGS.max_doc_len, feature_num]), [1,FLAGS.max_doc_len,1,1])
    ##
    tmp = tf.cast(range(FLAGS.max_doc_len), tf.int32)
    abs_cause = tf.tile(tf.reshape(tmp, [1, FLAGS.max_doc_len]), [FLAGS.max_doc_len, 1])
    abs_emo = tf.tile(tf.reshape(tmp, [FLAGS.max_doc_len, 1]), [1, FLAGS.max_doc_len])
    relative_pos = tf.nn.embedding_lookup(pos_embedding, abs_cause - abs_emo + 100)
    relative_pos = tf.tile(tf.reshape(relative_pos, [1, FLAGS.max_doc_len, FLAGS.max_doc_len, feature_num]), [tf.shape(s_emo)[0],1,1,1])
    ##
    pair_x = tf.nn.relu(s_emo + s_cause + pred_emo_feature + pred_cause_feature + relative_pos)
    mask = tf.cast(tf.sequence_mask(doc_len, FLAGS.max_doc_len), tf.float32)
    mask = tf.expand_dims(tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2), 3)
    pair_x = pair_x * mask
    # [batch, FLAGS.max_doc_len, FLAGS.max_doc_len, feature_num])
    print('pair features Done!')

    ########################### pair interaction & prediction ########################################################################
    print('pair interaction')
    for i in range(FLAGS.trans_iter):
        pair_x = CR_2Dtrans(pair_x, n_hidden = feature_num, n_head = 1, scope="CR_2Dtrans{}".format(i))
    print('pair interaction Done!')

    pred_pair, reg_tmp = pair_prediction(pair_x, feature_num, scope="pair_prediction")
    pred_pair = tf.reshape(pred_pair, [-1, FLAGS.max_doc_len * FLAGS.max_doc_len, FLAGS.n_class])
    return pred_pair, reg_tmp

def build_model(word_embedding, pos_embedding, x, sen_len, doc_len, is_training):
    x = tf.nn.embedding_lookup(word_embedding, x)
    x = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    x = tf.nn.dropout(x, keep_prob = is_training * FLAGS.keep_prob1 + (1.-is_training))
    sen_len = tf.reshape(sen_len, [-1])
    # x shape:        [-1, FLAGS.max_sen_len, FLAGS.embedding_dim]

    ########################################## emotion & cause extraction  ############
    print('building subtasks')
    pred_emo, pred_cause, s_emo, s_cause, reg = build_subtasks(x, sen_len, doc_len, is_training)
    print('build subtasks Done!')
    feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
    pred_emo_feature = tf.stop_gradient(pred_emo * feature_mask + 1e-8)
    pred_cause_feature = tf.stop_gradient(pred_cause * feature_mask + 1e-8)

    ########################################## emotion-cause pair extraction  ############
    if FLAGS.trans_type=='cross_road':
        pred_pair, reg_tmp = build_maintask_CR(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding, doc_len)
    else:
        pred_pair, reg_tmp = build_maintask_WC(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding)
    reg += reg_tmp
        
    return pred_emo, pred_cause, pred_pair, reg

def print_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('model_type {} \ntrans_type {} \ntrans_iter {} \nwindow_size {}'.format(
        FLAGS.model_type,  FLAGS.trans_type, FLAGS.trans_iter, FLAGS.window_size))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('batch {} \nlr {} \nkb1 {} \nkb2 {} \nl2_reg {}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('FLAGS.emo {} \nFLAGS.cause {} \nFLAGS.pair {} \nthreshold {} \ntraining_iter {}\n\n'.format(
        FLAGS.emo,  FLAGS.cause, FLAGS.pair, FLAGS.threshold, FLAGS.training_iter))


def get_batch_data(x, sen_len, doc_len, is_training, y_emotion, y_cause, y_pair, batch_size, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], is_training, y_emotion[index], y_cause[index], y_pair[index]]
        yield feed_list, len(index)

def run():
    if FLAGS.log_file_name:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open(FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 'data/all_data_pair.txt', FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.float32) # for Bert
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    if FLAGS.trans_type=='cross_road':
        y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * FLAGS.max_doc_len, FLAGS.n_class])
    else:
        y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, is_training, y_emotion, y_cause, y_pair]
    
    pred_emo, pred_cause, pred_pair, reg = build_model(word_embedding, pos_embedding, x, sen_len, doc_len, is_training)
    print('build model done!\n')

    loss_emo = - tf.reduce_sum(y_emotion * tf.log(pred_emo)) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
    loss_cause = - tf.reduce_sum(y_cause * tf.log(pred_cause)) / tf.cast(tf.reduce_sum(y_cause), dtype=tf.float32)
    loss_pair = - tf.reduce_sum(y_pair * tf.log(pred_pair)) / tf.cast(tf.reduce_sum(y_pair), dtype=tf.float32)
    loss_op = loss_cause * FLAGS.cause + loss_emo * FLAGS.emo + loss_pair * FLAGS.pair + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    
    true_y_emo_op = tf.argmax(y_emotion, 2)
    pred_y_emo_op = tf.argmax(pred_emo, 2)
    true_y_cause_op = tf.argmax(y_cause, 2)
    pred_y_cause_op = tf.argmax(pred_cause, 2)
    true_y_pair_op = y_pair
    pred_y_pair_op = pred_pair
    
    # Training Code Block
    print_info()
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        emo_list, cause_list, pair_list = [], [], []
        
        for fold in range(1,11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            
            #data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold)
            test_file_name = 'fold{}_test.txt'.format(fold)
            if FLAGS.trans_type=='cross_road':
                tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len = load_data_CR('data/'+train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
                te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x, te_sen_len, te_doc_len = load_data_CR('data/'+test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            else:
                tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len, tr_pair_left_cnt = load_data_WC('data/'+train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len, window_size = FLAGS.window_size)
                te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x, te_sen_len, te_doc_len, te_pair_left_cnt = load_data_WC('data/'+test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len, window_size = FLAGS.window_size)
            
            max_f1_emo, max_f1_cause, max_f1_pair = [-1.] * 3
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
            for i in xrange(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, 1., tr_y_emotion, tr_y_cause, tr_y_pair, FLAGS.batch_size):
                    _, loss, pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, true_y_emo_op, pred_y_pair_op, true_y_pair_op, doc_len], feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss))
                        p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                        print('emotion_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                        p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                        print('cause_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                        if FLAGS.trans_type=='cross_road':
                            p, r, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, threshold = FLAGS.threshold)
                        else:
                            p, r, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                        print('pair_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    step = step + 1
                # test
                test = [te_x, te_sen_len, te_doc_len, 0., te_y_emotion, te_y_cause, te_y_pair]
                loss, pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch = sess.run(
                        [loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, true_y_emo_op, pred_y_pair_op, true_y_pair_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time()-start_time))

                p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                if f1 > max_f1_emo:
                    max_p_emo, max_r_emo, max_f1_emo = p, r, f1
                print('emotion_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_emo, max_r_emo, max_f1_emo))

                p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                if f1 > max_f1_cause:
                    max_p_cause, max_r_cause, max_f1_cause = p, r, f1
                print('cause_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_cause, max_r_cause, max_f1_cause))

                if FLAGS.trans_type=='cross_road':
                    p, r, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, threshold = FLAGS.threshold)
                else:
                    p, r, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, te_pair_left_cnt, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                if f1 > max_f1_pair:
                    max_p_pair, max_r_pair, max_f1_pair = p, r, f1
                print('pair_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_pair, max_r_pair, max_f1_pair))

            print('Optimization Finished!\n')
            print('############# fold {} end ###############'.format(fold))
            
            emo_list.append([max_p_emo, max_r_emo, max_f1_emo])
            cause_list.append([max_p_cause, max_r_cause, max_f1_cause])
            pair_list.append([max_p_pair, max_r_pair, max_f1_pair])
            
              
        emo_list, cause_list, pair_list = map(lambda x: np.array(x), [emo_list, cause_list, pair_list])

        print('\nemotion_prediction: test f1 in 10 fold: {}'.format(emo_list[:,2:]))
        p, r, f1 = emo_list.mean(axis=0)
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        print('\ncause_prediction: test f1 in 10 fold: {}'.format(cause_list[:,2:]))
        p, r, f1 = cause_list.mean(axis=0)
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        print('\npair_prediction: test f1 in 10 fold: {}'.format(pair_list[:,2:]))
        p, r, f1 = pair_list.mean(axis=0)
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        print_time()
     
def main(_):
    for FLAGS.model_type in ['Indep', 'Inter-EC', 'Inter-CE']:
        FLAGS.trans_type, FLAGS.trans_iter = 'cross_road', 0
        # FLAGS.log_file_name = 'log/ECPE-2D({})_1.log'.format(FLAGS.model_type)
        run()

        FLAGS.trans_type, FLAGS.trans_iter = 'window_constrained', 1
        # FLAGS.log_file_name = 'log/ECPE-2D({}+WC)_trans_iter{}_1.log'.format(FLAGS.model_type, FLAGS.trans_iter)
        run()

        FLAGS.trans_type, FLAGS.trans_iter = 'cross_road', 2
        # FLAGS.log_file_name = 'log/ECPE-2D({}+CR)_trans_iter{}_1.log'.format(FLAGS.model_type, FLAGS.trans_iter)
        run()
    

if __name__ == '__main__':
    tf.app.run() 