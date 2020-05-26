import os
import tensorflow as tf
import sys
from dataloader import *
from rec import *
from ubr import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 500

# for TMALL
FEAT_SIZE_TMALL = 1529672 + 6 #(6 is for time context)
DATA_DIR_TMALL = '../data/tmall/feateng_data/'

# for CCMR
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1) + 4
DATA_DIR_CCMR = '../data/ccmr/feateng_data/'

# for TAOBAO
FEAT_SIZE_TAOBAO = 5062314
DATA_DIR_TAOBAO = '../data/taobao/feateng_data/'

# for ALIPAY
FEAT_SIZE_ALIPAY = 2836410
DATA_DIR_ALIPAY = '../../ubr4rec-data/alipay/feateng_data/'

def restore(data_set_name, target_test_file, user_feat_dict_file, item_feat_dict_file, context_dict_file,
        rec_model_type, ubr_model_type, b_num, train_batch_size, feature_size, eb_dim, hidden_size, 
        rec_lr, ubr_lr, reg_lambda, record_fnum, emb_initializer, taker):
    print('restore begin')
    tf.reset_default_graph()

    if rec_model_type == 'RecSum':
        rec_model = RecSum(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
    elif rec_model_type == 'RecAtt':
        rec_model = RecAtt(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
    else:
        print('WRONG REC MODEL TYPE')
        exit(1)
    
    if ubr_model_type == 'UBR_SA':
        ubr_model = UBR_SA(feature_size, eb_dim, hidden_size, record_fnum, emb_initializer)
    else:
        print('WRONG UBR MODEL TYPE')
        exit(1)

    rec_model_name = '{}_{}_{}_{}'.format(rec_model_type, train_batch_size, rec_lr, reg_lambda)
    ubr_model_name = '{}_{}_{}'.format(ubr_model_type, train_batch_size, ubr_lr)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        rec_model.restore(sess, 'save_model_{}/{}/{}_{}/{}/ckpt'.format(data_set_name, b_num, rec_model_type, ubr_model_type, rec_model_name))
        ubr_model.restore(sess, 'save_model_{}/{}/{}_{}/{}/ckpt'.format(data_set_name, b_num, rec_model_type, ubr_model_type, ubr_model_name))
        print('restore eval begin')
        _, logloss, rig, auc  = eval(rec_model, ubr_model, sess, target_test_file, user_feat_dict_file, item_feat_dict_file, context_dict_file, reg_lambda, taker, train_batch_size)
        
        print('RESTORE, LOGLOSS %.4f  RIG: %.4f  AUC: %.4f' % (logloss, rig, auc))
        with open('logs_{}/{}/{}_{}/{}.txt'.format(data_set_name, b_num, rec_model_type, ubr_model_type, rec_model_type), 'a') as f:
            results = [train_batch_size, rec_lr, reg_lambda, logloss, rig, auc]
            results = [rec_model_type] + [str(res) for res in results]
            result_line = '\t'.join(results) + '\n'
            f.write(result_line)

def eval(rec_model, ubr_model, sess, target_file, user_feat_dict_file, item_feat_dict_file, 
        context_dict_file, reg_lambda, taker, batch_size):
    preds = []
    labels = []
    losses = []
    
    data_loader = DataLoader_Target(batch_size, target_file, user_feat_dict_file, item_feat_dict_file, context_dict_file)
    
    t = time.time()
    for batch_data in data_loader:
        target_batch, label_batch = batch_data
        index_batch = ubr_model.get_index(sess, target_batch)
        seq_batch, seq_len_batch = taker.take_behave(target_batch, index_batch)

        pred, label, loss = rec_model.eval(sess, [seq_batch, seq_len_batch, target_batch, label_batch], reg_lambda)
        preds += pred
        labels += label
        losses.append(loss)

    logloss = log_loss(labels, preds)
    rig = 1 -(logloss / -(0.5 * math.log(0.5) + (1 - 0.5) * math.log(1 - 0.5)))
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, logloss, rig, auc

def train_rec_model(rec_training_monitor, epoch_num, sess, eval_iter_num, train_batch_size, 
                    taker, lr, reg_lambda, rec_model, ubr_model, 
                    target_train_file, user_feat_dict_file,
                    item_feat_dict_file, context_dict_file, step, b_num):
    early_stop = False
    losses_step = []
    auc_step = []
    logloss_step = []
    rig_step = []

    for epoch in range(epoch_num):
        if early_stop:
            break
        # train rec model
        data_loader = DataLoader_Target(train_batch_size, target_train_file, user_feat_dict_file, 
                                item_feat_dict_file, context_dict_file)
        t = time.time()
        for batch_data in data_loader:
            if early_stop:
                break
            # get the retrieve data
            target_batch, label_batch = batch_data
            index_batch = ubr_model.get_index(sess, target_batch)
            seq_batch, seq_len_batch = taker.take_behave(target_batch, index_batch)
            new_batch_data = [seq_batch, seq_len_batch, target_batch, label_batch]

            # run train and eval
            loss = rec_model.train(sess, new_batch_data, lr, reg_lambda)
            pred, label, _ = rec_model.eval(sess, new_batch_data, reg_lambda)
            step += 1

            # calculate evaluation metrics
            logloss = log_loss(label, pred)
            rig = 1 -(logloss / -(0.5 * math.log(0.5) + (1 - 0.5) * math.log(1 - 0.5)))
            auc = roc_auc_score(label, pred)
            losses_step.append(loss)
            auc_step.append(auc)
            logloss_step.append(logloss)
            rig_step.append(rig)
            # print evaluation results
            if step % eval_iter_num == 0:
                train_loss = sum(losses_step) / len(losses_step)
                rec_training_monitor['loss'].append(train_loss)
                losses_step = []
                
                train_auc = sum(auc_step) / len(auc_step)
                rec_training_monitor['auc'].append(train_auc)
                auc_step = []

                train_logloss = sum(logloss_step) / len(logloss_step)
                rec_training_monitor['logloss'].append(train_logloss)
                logloss_step = []

                train_rig = sum(rig_step) / len(rig_step)
                rec_training_monitor['rig'].append(train_rig)
                rig_step = []

                print("TIME UNTIL EVAL: %.4f" % (time.time() - t))
                print("REC MODEL STEP %d  LOSS: %.4f  LOGLOSS: %.4f  RIG: %.4f  AUC: %.4f" % (step, train_loss, train_logloss, train_rig, train_auc))
                t = time.time()
                
                if len(rec_training_monitor['auc']) >= 2:
                    if rec_training_monitor['auc'][-1] > max(rec_training_monitor['auc'][:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(rec_model_type, train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/{}_{}/{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name)):
                            os.makedirs('save_model_{}/{}/{}_{}/{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name))
                        save_path = 'save_model_{}/{}/{}_{}/{}/ckpt'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name)
                        rec_model.save(sess, save_path)
                
                if len(rec_training_monitor['loss']) > 2:
                    if (rec_training_monitor['loss'][-1] > rec_training_monitor['loss'][-2] and rec_training_monitor['loss'][-2] > rec_training_monitor['loss'][-3]):
                        early_stop = True
                    if (rec_training_monitor['loss'][-2] - rec_training_monitor['loss'][-1]) <= 0.001 and (rec_training_monitor['loss'][-3] - rec_training_monitor['loss'][-2]) <= 0.001:
                        early_stop = True
    return step, early_stop

def train_ubr_model(ubr_training_monitor, epoch_num, sess, eval_iter_num, taker, lr, 
                    train_batch_size, rec_model, ubr_model, target_train_file, 
                    user_feat_dict_file, item_feat_dict_file, context_dict_file, 
                    summary_writer, step, b_num):
    loss_step = []
    reward_step = []
    
    for i in range(epoch_num):
        data_loader = DataLoader_Target(train_batch_size, target_train_file, user_feat_dict_file, 
                                item_feat_dict_file, context_dict_file)

        t = time.time()
        i = 0
        for batch_data in data_loader:
            target_batch, label_batch = batch_data
            index_batch = ubr_model.get_index(sess, target_batch)
            seq_batch, seq_len_batch = taker.take_behave(target_batch, index_batch)
            new_batch_data = [seq_batch, seq_len_batch, target_batch, label_batch]

            rewards = rec_model.get_reward(sess, new_batch_data)
            loss, reward, summary = ubr_model.train(sess, target_batch, lr, rewards)
            loss_step.append(loss)
            reward_step.append(reward)

            summary_writer.add_summary(summary, step)
            step += 1

            if step % eval_iter_num == 0:
                avg_loss = sum(loss_step) / len(loss_step)
                avg_reward = sum(reward_step) / len(reward_step)
                ubr_training_monitor['loss'].append(avg_loss)
                ubr_training_monitor['reward'].append(avg_reward)
                loss_step = []
                reward_step = []
                
                print("TIME UNTIL EVAL: %.4f" % (time.time() - t))
                print("UBR MODEL STEP %d  LOSS: %.4f  REWARD: %.4f" % (step, avg_loss, avg_reward))
                t = time.time()

    # save model
    model_name = '{}_{}_{}'.format(ubr_model_type, train_batch_size, lr)
    if not os.path.exists('save_model_{}/{}/{}_{}/{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name)):
        os.makedirs('save_model_{}/{}/{}_{}/{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name))
    save_path = 'save_model_{}/{}/{}_{}/{}/ckpt'.format(data_set_name, b_num, rec_model_type, ubr_model_type, model_name)
    ubr_model.save(sess, save_path)
    return step

def train(data_set_name, target_train_file, user_feat_dict_file,
        item_feat_dict_file, context_dict_file, rec_model_type, ubr_model_type,
        taker, b_num, train_batch_size, feature_size, eb_dim, hidden_size, 
        rec_lr, ubr_lr, reg_lambda, dataset_size, record_fnum, emb_initializer):
    tf.reset_default_graph()

    if rec_model_type == 'RecSum':
        rec_model = RecSum(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
    elif rec_model_type == 'RecAtt':
        rec_model = RecAtt(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
    else:
        print('WRONG REC MODEL TYPE')
        exit(1)
    
    if ubr_model_type == 'UBR_SA':
        ubr_model = UBR_SA(feature_size, eb_dim, hidden_size, record_fnum, emb_initializer)
    else:
        print('WRONG UBR MODEL TYPE')
        exit(1)
    
    rec_training_monitor = {
        'loss' : [],
        'logloss' : [],
        'rig' : [],
        'auc' : []
    }

    ubr_training_monitor = {
        'loss' : [],
        'reward' : []
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        rec_step = 0
        ubr_step = 0
        eval_iter_num = (dataset_size // 25) // batch_size
        
        # summary writer
        if not os.path.exists('summary_{}/{}/{}_{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type)):
            os.makedirs('summary_{}/{}/{}_{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type))
        rec_model_name = '{}_{}_{}_{}'.format(rec_model_type, batch_size, rec_lr, reg_lambda)
        ubr_model_name = '{}_{}_{}'.format(ubr_model_type, batch_size, ubr_lr)
        summary_writer_ubr = tf.summary.FileWriter('summary_{}/{}/{}_{}/{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type, ubr_model_name))

        # begin training process
        rec_step, early_stop = train_rec_model(rec_training_monitor, 1, sess, eval_iter_num, train_batch_size,
                                            taker, rec_lr, reg_lambda, rec_model, ubr_model, target_train_file, user_feat_dict_file,
                                            item_feat_dict_file, context_dict_file, rec_step, b_num)
        for i in range(10):
            ubr_step = train_ubr_model(ubr_training_monitor, 1, sess, eval_iter_num, taker, ubr_lr, train_batch_size,
                                        rec_model, ubr_model, target_train_file, user_feat_dict_file,
                                        item_feat_dict_file, context_dict_file, summary_writer_ubr, ubr_step, b_num)

            rec_step, early_stop = train_rec_model(rec_training_monitor, 1, sess, eval_iter_num, train_batch_size,
                                            taker, rec_lr, reg_lambda, rec_model, ubr_model, target_train_file, user_feat_dict_file,
                                            item_feat_dict_file, context_dict_file, rec_step, b_num)
            if early_stop:
                break

        # generate log
        if not os.path.exists('logs_{}/{}/{}_{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type)):
            os.makedirs('logs_{}/{}/{}_{}/'.format(data_set_name, b_num, rec_model_type, ubr_model_type))
        
        with open('logs_{}/{}/{}_{}/{}.pkl'.format(data_set_name, b_num, rec_model_type, ubr_model_type, rec_model_name), 'wb') as f:
            pkl.dump(rec_training_monitor, f)
        with open('logs_{}/{}/{}_{}/{}.pkl'.format(data_set_name, b_num, rec_model_type, ubr_model_type, ubr_model_name), 'wb') as f:
            pkl.dump(ubr_training_monitor, f)
        
        
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [REC MODEL TYPE] [UBR MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    rec_model_type = sys.argv[1]
    ubr_model_type = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
    data_set_name = sys.argv[4]

    if data_set_name == 'tmall':
        record_fnum = 9

        target_train_file = DATA_DIR_TMALL + 'target_train.txt'
        target_test_file = DATA_DIR_TMALL + 'target_test.txt'

        user_feat_dict_file = DATA_DIR_TMALL + 'user_feat_dict.pkl'
        item_feat_dict_file = DATA_DIR_TMALL + 'item_feat_dict.pkl'
        context_dict_train_file = DATA_DIR_TMALL + 'context_dict_train.pkl'
        context_dict_test_file = DATA_DIR_TMALL + 'context_dict_test.pkl'

        # model parameter
        feature_size = FEAT_SIZE_TMALL
        dataset_size = 847568

        emb_initializer = None
        b_num = 20
        reader = ESReader('tmall')


    elif data_set_name == 'taobao':
        record_fnum = 4

        target_train_file = DATA_DIR_TAOBAO + 'target_train.txt'
        target_test_file = DATA_DIR_TAOBAO + 'target_test.txt'

        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_TAOBAO + 'item_feat_dict.pkl'
        context_dict_train_file = DATA_DIR_TAOBAO + 'context_dict_train.pkl'
        context_dict_test_file = DATA_DIR_TAOBAO + 'context_dict_test.pkl'

        # model parameter
        feature_size = FEAT_SIZE_TAOBAO
        dataset_size = 1962046

        emb_initializer = None
        b_num = 20
        reader = ESReader('taobao')

    elif data_set_name == 'alipay':
        record_fnum = 6

        target_train_file = DATA_DIR_ALIPAY + 'target_train.txt'
        target_test_file = DATA_DIR_ALIPAY + 'target_test.txt'

        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_ALIPAY + 'item_feat_dict.pkl'
        context_dict_train_file = DATA_DIR_ALIPAY + 'context_dict_train.pkl'
        context_dict_test_file = DATA_DIR_ALIPAY + 'context_dict_test.pkl'

        # model parameter
        feature_size = FEAT_SIZE_ALIPAY
        dataset_size = 996616

        emb_initializer = None
        b_num = 12
        reader = ESReader('alipay')
    else:
        print('WRONG DATASET NAME: {}'.format(data_set_name))
        exit()

    ################################## training hyper params ##################################
    
    reg_lambdas = [1e-4]
    batch_sizes = [100, 200]
    rec_lrs = [5e-4, 1e-3]
    ubr_lrs = [1e-6, 1e-5, 1e-4]

    for reg_lambda in reg_lambdas:
        for i in range(len(batch_sizes)):
            batch_size = batch_sizes[i]
            rec_lr = rec_lrs[i]
            taker = Taker(reader, batch_size, b_num, record_fnum)

            for ubr_lr in ubr_lrs:
                train(data_set_name, target_train_file, user_feat_dict_file,
                    item_feat_dict_file, context_dict_train_file, rec_model_type, ubr_model_type,
                    taker, b_num, batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, 
                    rec_lr, ubr_lr, reg_lambda, dataset_size, record_fnum, emb_initializer)
                restore(data_set_name, target_test_file, user_feat_dict_file, item_feat_dict_file, context_dict_test_file,
                        rec_model_type, ubr_model_type, b_num, batch_size, feature_size, 
                        EMBEDDING_SIZE, HIDDEN_SIZE, rec_lr, ubr_lr, reg_lambda, 
                        record_fnum, emb_initializer, taker)

