import pickle as pkl
import random
import numpy as np
import sys
from elastic_client import *
import datetime

RAW_DIR = '../ubr4rec-data/alipay/raw_data/'
FEATENG_DIR = '../ubr4rec-data/alipay/feateng_data/'

ORI_FEAT_SIZE = 2836404
FEAT_SIZE = ORI_FEAT_SIZE + 6
SECONDS_PER_DAY = 24 * 3600

def feateng(in_file, remap_dict_file):
    uid_remap_dict = {}
    iid_remap_dict = {}
    sid_remap_dict = {}
    cid_remap_dict = {}

    uid_set = set()
    iid_set = set()
    sid_set = set()
    cid_set = set()

    with open(in_file, 'r') as r:
        i = 0
        for line in r:
            if i == 0:
                i += 1
                continue
            uid, sid, iid, cid, btype, date = line[:-1].split(',')
            if btype == '0':
                uid_set.add(uid)
                iid_set.add(iid)
                sid_set.add(sid)
                cid_set.add(cid)

    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)
    sid_list = list(sid_set)

    print('user number is: {}'.format(len(uid_list)))        
    print('item number is: {}'.format(len(iid_list)))
    
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
    for sid in sid_list:
        sid_remap_dict[sid] = str(feature_id)
        feature_id += 1
    print('total original feature number: {}'.format(feature_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
        pkl.dump(sid_remap_dict, f)
    print('remap dict dumpped')

def get_season(month):
    if month >= 10:
        return 3
    elif month >= 7 and month <= 9:
        return 2
    elif month >= 4 and month <= 6:
        return 1
    else:
        return 0

def get_ud(day):
    if day <= 15:
        return 0
    else:
        return 1

def remap_log_file(input_log_file, remap_dict_file, output_log_file, item_feat_dict_file):
    with open(remap_dict_file, 'rb') as f:
        uid_remap_dict = pkl.load(f)
        iid_remap_dict = pkl.load(f)
        cid_remap_dict = pkl.load(f)
        sid_remap_dict = pkl.load(f)
    item_feat_dict = {}
    newlines = []

    with open(input_log_file, 'r') as f:
        for line in f:
            uid, sid, iid, cid, btype, date = line[:-1].split(',')
            if btype != '0':
                continue
            uid = uid_remap_dict[uid]
            iid = iid_remap_dict[iid]
            cid = cid_remap_dict[cid]
            sid = sid_remap_dict[sid]
            
            ts = str(int(time.mktime(datetime.datetime.strptime(date, "%Y%m%d").timetuple())))
            
            month = int(date[4:6])
            day = int(date[6:])
            sea_id = str(get_season(month) + ORI_FEAT_SIZE)
            ud_id = str(get_ud(day) + ORI_FEAT_SIZE + 4)
            
            if iid not in item_feat_dict:
                item_feat_dict[iid] = [cid, sid]
            
            newline = ','.join([uid, iid, cid, sid, sea_id, ud_id, ts]) + '\n'
            newlines.append(newline)
        
    with open(output_log_file, 'w') as f:
        f.writelines(newlines)
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)


def sort_raw_log(raw_log_ts_file, sorted_raw_log_ts_file):
    line_dict = {}
    with open(raw_log_ts_file) as f:
        for line in f:
            uid, _, _, _, _, _, ts = line[:-1].split(',')
            if uid not in line_dict:
                line_dict[uid] = [[line, int(ts)]]
            else:
                line_dict[uid].append([line, int(ts)])
            
    for uid in line_dict:
        line_dict[uid].sort(key = lambda x:x[1])
    print('sort complete')
    print(len(line_dict.keys()))
    newlines = []
    for uid in line_dict:
        for tup in line_dict[uid]:
            newlines.append(tup[0])
    with open(sorted_raw_log_ts_file, 'w') as f:
        f.writelines(newlines)


def random_sample(min = 626042, max = 2826332):
    return str(random.randint(min, max))

def neg_sample(user_seq):
    r = random.randint(0, 4)
    if r == 0:
        return random_sample()
    else:
        return random.choice(user_seq)

def gen_target_seq(input_file, 
                    target_train_file, 
                    target_vali_file, 
                    target_test_file, 
                    user_seq_file, 
                    database_file,
                    context_dict_train_file, 
                    context_dict_vali_file, 
                    context_dict_test_file):
    line_dict = {}
    user_seq_dict = {}
    context_dict_train = {}
    context_dict_vali = {}
    context_dict_test = {}


    with open(input_file, 'r') as f:
        for line in f:
            uid, iid, cid, sid, sea_id, ud_id, ts = line[:-1].split(',')
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
                target_train_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3])]) + '\n']
                context_dict_train[uid] = list(map(int, line_dict[uid][-3][:-1].split(',')[-3:-1]))

                target_vali_lines += [','.join([uid, user_seq_dict[uid][-2]]) + '\n']
                target_vali_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3])]) + '\n']
                context_dict_vali[uid] = list(map(int, line_dict[uid][-2][:-1].split(',')[-3:-1]))

                target_test_lines += [','.join([uid, user_seq_dict[uid][-1]]) + '\n']
                target_test_lines += [','.join([uid, neg_sample(user_seq_dict[uid][:-3])]) + '\n']
                context_dict_test[uid] = list(map(int, line_dict[uid][-1][:-1].split(',')[-3:-1]))
                
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
    writer = ESWriter(input_file, 'alipay')
    writer.write()

if __name__ == "__main__":
    feateng(RAW_DIR + 'ijcai2016_taobao.csv', FEATENG_DIR + 'id_remap_dict.pkl')
    remap_log_file(RAW_DIR + 'ijcai2016_taobao.csv', FEATENG_DIR + 'id_remap_dict.pkl', FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'item_feat_dict.pkl')
    sort_raw_log(FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'sorted_remapped_log.csv')
    gen_target_seq(FEATENG_DIR + 'sorted_remapped_log.csv',
                    FEATENG_DIR + 'target_train.txt', FEATENG_DIR + 'target_vali.txt', FEATENG_DIR + 'target_test.txt', FEATENG_DIR + 'user_seq.txt', FEATENG_DIR + 'database.txt', 
                    FEATENG_DIR + 'context_dict_train.pkl', FEATENG_DIR + 'context_dict_vali.pkl', FEATENG_DIR + 'context_dict_test.pkl')
    insert_elastic(FEATENG_DIR + 'database.txt')
