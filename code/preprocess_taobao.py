import pickle as pkl
import random
import numpy as np
import sys
from elastic_client import *

RAW_DIR = '../data/taobao/raw_data/'
FEATENG_DIR = '../data/taobao/feateng_data/'

ORI_FEAT_SIZE = 5062312
FEAT_SIZE = ORI_FEAT_SIZE + 2
START_TIME = 1511539200
SECONDS_PER_DAY = 24 * 3600

def feateng(in_file, remap_dict_file):
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}

    uid_set = set()
    iid_set = set()
    cid_set = set()

    with open(in_file, 'r') as r:
        for line in r:
            uid, iid, cid, btype, ts = line.split(',')
            if btype == 'pv':
                uid_set.add(uid)
                iid_set.add(iid)
                cid_set.add(cid)
            
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)
    

    feature_id = 1
    for uid in uid_list:
        uid_remap_dict[uid] = str(feature_id)
        feature_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(feature_id)
        feature_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(feature_id)
        feature_id += 1
    print('total original feature number: {}'.format(feature_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
    print('remap dict dumpped')

def isweekday(date):
    if date in [0, 1, 8]:
        return str(ORI_FEAT_SIZE)
    else:
        return str(ORI_FEAT_SIZE + 1)

def remap_log_file(input_log_file, remap_dict_file, output_log_file, user_seq_dict_file, item_feat_dict_file):
    with open(remap_dict_file, 'rb') as f:
        uid_remap_dict = pkl.load(f)
        iid_remap_dict = pkl.load(f)
        cid_remap_dict = pkl.load(f)
    user_seq_dict = {}
    item_feat_dict = {}
    newlines = []

    with open(input_log_file, 'r') as f:
        for line in f:
            uid, iid, cid, btype, ts = line[:-1].split(',')
            if btype != 'pv':
                continue
            uid = uid_remap_dict[uid]
            iid = iid_remap_dict[iid]
            cid = cid_remap_dict[cid]
            
            date = (int(ts) - START_TIME) // SECONDS_PER_DAY
            if date < 0:
                continue
            date = isweekday(date)
            
            if uid not in user_seq_dict:
                user_seq_dict[uid] = [iid]
            else:
                user_seq_dict[uid].append(iid)
            
            if iid not in item_feat_dict:
                item_feat_dict[iid] = [cid]
            
            newline = ','.join([uid, iid, cid, date, ts]) + '\n'
            newlines.append(newline)
        
    with open(output_log_file, 'w') as f:
        f.writelines(newlines)
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)
    with open(user_seq_dict_file, 'wb') as f:
        pkl.dump(user_seq_dict, f)

def neg_sample(user_seq, items):
    r = random.randint(0, 1)
    if r == 1:
        return random.choice(user_seq)
    else:
        return random.choice(items)

def gen_target_seq(input_file, 
                    item_feat_dict_file,
                    target_train_file, 
                    target_vali_file, 
                    target_test_file, 
                    user_seq_file, 
                    database_file,
                    context_dict_train_file, 
                    context_dict_vali_file, 
                    context_dict_test_file):
    with open(item_feat_dict_file, 'rb') as f:
        d = pkl.load(f)
    items = []
    for item in d.keys():
        items.append(item)

    line_dict = {}
    user_seq_dict = {}
    context_dict_train = {}
    context_dict_vali = {}
    context_dict_test = {}

    with open(input_file, 'r') as f:
        for line in f:
            uid, iid, cid, did, time_stamp = line[:-1].split(',')
            if uid not in line_dict:
                line_dict[uid] = [line]
                user_seq_dict[uid] = [iid]
            else:
                line_dict[uid].append(line)
                user_seq_dict[uid].append(iid)
        

        target_train_lines = []
        target_vali_lines = []
        target_test_lines = []
        user_seq_lines = []
        database_lines = []
        
        for uid in user_seq_dict:
            if len(user_seq_dict[uid]) > 3:
                target_train_lines += [','.join([uid, user_seq_dict[uid][-3]]) + '\n']
                target_train_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3], items)]) + '\n']
                context_dict_train[uid] = [int(line_dict[uid][-3][:-1].split(',')[-2])]

                target_vali_lines += [','.join([uid, user_seq_dict[uid][-2]]) + '\n']
                target_vali_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3], items)]) + '\n']
                context_dict_vali[uid] = [int(line_dict[uid][-2][:-1].split(',')[-2])]

                target_test_lines += [','.join([uid, user_seq_dict[uid][-1]]) + '\n']
                target_test_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3], items)]) + '\n']
                context_dict_test[uid] = [int(line_dict[uid][-1][:-1].split(',')[-2])]
                
                user_seq = user_seq_dict[uid][:-3]
                user_seq_lines += [','.join(user_seq) + '\n'] * 2 #(1 pos and 1 neg item)
                
                database_lines += line_dict[uid][:-3]
        
        with open(target_train_file, 'w') as f:
            f.writelines(target_train_lines)
        with open(target_vali_file, 'w') as f:
            f.writelines(target_vali_lines)
        with open(target_test_file, 'w') as f:
            f.writelines(target_test_lines)
        
        with open(user_seq_file, 'w') as f:
            f.writelines(user_seq_lines)
        with open(database_file, 'w') as f:
            f.writelines(database_lines)
        
        with open(context_dict_train_file, 'wb') as f:
            pkl.dump(context_dict_train, f)
        with open(context_dict_vali_file, 'wb') as f:
            pkl.dump(context_dict_vali, f)
        with open(context_dict_test_file, 'wb') as f:
            pkl.dump(context_dict_test, f)


def insert_elastic(input_file):      
    writer = ESWriter(input_file, 'taobao')
    writer.write()


if __name__ == "__main__":
    feateng(RAW_DIR + 'UserBehavior.csv', FEATENG_DIR + 'id_remap_dict.pkl')
    remap_log_file(RAW_DIR + 'UserBehavior.csv', FEATENG_DIR + 'id_remap_dict.pkl', FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'user_seq_dict.pkl', FEATENG_DIR + 'item_feat_dict.pkl')
    gen_target_seq(FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'item_feat_dict.pkl', FEATENG_DIR + 'target_train.txt', FEATENG_DIR + 'target_vali.txt', FEATENG_DIR + 'target_test.txt', FEATENG_DIR + 'user_seq.txt', FEATENG_DIR + 'database.txt', 
                    FEATENG_DIR + 'context_dict_train.pkl', FEATENG_DIR + 'context_dict_vali.pkl', FEATENG_DIR + 'context_dict_test.pkl')
    insert_elastic(FEATENG_DIR + 'database.txt')
    
