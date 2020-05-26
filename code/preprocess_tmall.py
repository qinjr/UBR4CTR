import pickle as pkl
import random
import numpy as np
import datetime
import time
import sys
from elastic_client import *

random.seed(1111)

RAW_DIR = '../data/tmall/raw_data/'
FEATENG_DIR = '../data/tmall/feateng_data/'
ORI_FEATSIZE = 1529672

def join_user_profile(user_profile_file, behavior_file, joined_file):
    user_profile_dict = {}
    with open(user_profile_file, 'r') as f:
        for line in f:
            uid, aid, gid = line[:-1].split(',')
            user_profile_dict[uid] = ','.join([aid, gid])
    
    # join
    newlines = []
    with open(behavior_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            user_profile = user_profile_dict[uid]
            newlines.append(line[:-1] + ',' + user_profile + '\n')
    with open(joined_file, 'w') as f:
        f.writelines(newlines)


def feateng(joined_raw_file, remap_dict_file, user_feat_dict_file, item_feat_dict_file):
    uid_set = set()
    iid_set = set()
    cid_set = set()
    sid_set = set()
    bid_set = set()
    aid_set = set()
    gid_set = set()
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, iid, cid, sid, bid, date_str, btypeid, aid, gid = line[:-1].split(',')
            uid_set.add(uid)
            iid_set.add(iid)
            cid_set.add(cid)
            sid_set.add(sid)
            bid_set.add(bid)
            aid_set.add(aid)
            gid_set.add(gid)
            date_str = '2015' + date_str
            time_int = int(time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple()))
            t_idx = (time_int - START_TIME) // TIME_DELTA
            time_idxs.append(t_idx)

    # remap
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)
    sid_list = list(sid_set)
    bid_list = list(bid_set)
    aid_list = list(aid_set)
    gid_list = list(gid_set)

    print('user num: {}'.format(len(uid_list)))
    print('item num: {}'.format(len(iid_list)))
    print('cate num: {}'.format(len(cid_list)))
    print('seller num: {}'.format(len(sid_list)))
    print('brand num: {}'.format(len(bid_list)))
    print('age num: {}'.format(len(aid_list)))
    print('gender num: {}'.format(len(gid_list)))
    
    remap_id = 1
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}
    sid_remap_dict = {}
    bid_remap_dict = {}
    aid_remap_dict = {}
    gid_remap_dict = {}

    for uid in uid_list:
        uid_remap_dict[uid] = str(remap_id)
        remap_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(remap_id)
        remap_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(remap_id)
        remap_id += 1
    for sid in sid_list:
        sid_remap_dict[sid] = str(remap_id)
        remap_id += 1
    for bid in bid_list:
        bid_remap_dict[bid] = str(remap_id)
        remap_id += 1
    for aid in aid_list:
        aid_remap_dict[aid] = str(remap_id)
        remap_id += 1
    for gid in gid_list:
        gid_remap_dict[gid] = str(remap_id)
        remap_id += 1
    print('feat size: {}'.format(remap_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
        pkl.dump(sid_remap_dict, f)
        pkl.dump(bid_remap_dict, f)
        pkl.dump(aid_remap_dict, f)
        pkl.dump(gid_remap_dict, f)
    print('remap ids completed')

    # remap file generate
    item_feat_dict = {}
    user_feat_dict = {}
    # for dummy user
    user_feat_dict['0'] = [0, 0]
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            uid, iid, cid, sid, bid, time_stamp, btypeid, aid, gid = lines[i][:-1].split(',')
            uid_remap = uid_remap_dict[uid]
            iid_remap = iid_remap_dict[iid]
            cid_remap = cid_remap_dict[cid]
            sid_remap = sid_remap_dict[sid]
            bid_remap = bid_remap_dict[bid]
            aid_remap = aid_remap_dict[aid]
            gid_remap = gid_remap_dict[gid]
            item_feat_dict[iid_remap] = [int(cid_remap), int(sid_remap), int(bid_remap)]
            user_feat_dict[uid_remap] = [int(aid_remap), int(gid_remap)]
    print('remaped file generated')


    with open(user_feat_dict_file, 'wb') as f:
        pkl.dump(user_feat_dict, f)
    print('user feat dict dump completed')
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)
    print('item feat dict dump completed')

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
    

def remap(raw_file, remap_dict_file, remap_file):
    with open(remap_dict_file, 'rb') as f:
        uid_remap_dict = pkl.load(f)
        iid_remap_dict = pkl.load(f)
        cid_remap_dict = pkl.load(f)
        sid_remap_dict = pkl.load(f)
        bid_remap_dict = pkl.load(f)
        aid_remap_dict = pkl.load(f)
        gid_remap_dict = pkl.load(f)
    
    newlines = []
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, iid, cid, sid, bid, date, _, aid, gid = line[:-1].split(',')
            uid = uid_remap_dict[uid]
            iid = iid_remap_dict[iid]
            cid = cid_remap_dict[cid]
            sid = sid_remap_dict[sid]
            bid = bid_remap_dict[bid]
            aid = aid_remap_dict[aid]
            gid = gid_remap_dict[gid]
            
            month = int(date[:2])
            day = int(date[2:])
            sea_id = str(get_season(month) + ORI_FEATSIZE)
            ud_id = str(get_ud(day) + ORI_FEATSIZE + 4)

            date = '2015' + date
            time_stamp = str(int(time.mktime(datetime.datetime.strptime(date, "%Y%m%d").timetuple())))
            newline = ','.join([uid, aid, gid, iid, cid, sid, bid, sea_id, ud_id, time_stamp]) + '\n'
            newlines.append(newline)
    
    with open(remap_file, 'w') as f:
        f.writelines(newlines)

def sort_log(log_ts_file, sorted_log_ts_file):
    line_dict = {}
    with open(log_ts_file) as f:
        for line in f:
            line_items = line[:-1].split(',')
            uid = line_items[0]
            ts = int(line_items[-1])
            if uid not in line_dict:
                line_dict[uid] = [[line, ts]]
            else:
                line_dict[uid].append([line, ts])
            
    for uid in line_dict:
        line_dict[uid].sort(key = lambda x:x[1])
    print('sort complete')
    newlines = []
    for uid in line_dict:
        for tup in line_dict[uid]:
            newlines.append(tup[0])
    with open(sorted_log_ts_file, 'w') as f:
        f.writelines(newlines)


def random_sample(min = 424171, max = 1514560):
    return str(random.randint(min, max))

def neg_sample(user_seq):
    r = random.randint(0, 1)
    if r == 1:
        return random.choice(user_seq)
    else:
        return random_sample()

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
            uid, aid, gid, iid, cid, sid, bid, sea_id, ud_id, time_stamp = line[:-1].split(',')
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


if __name__ == "__main__":
    join_user_profile(RAW_DIR + 'user_info_format1.csv', RAW_DIR + 'user_log_format1.csv', FEATENG_DIR + 'joined_user_behavior.csv')
    feateng(FEATENG_DIR + 'joined_user_behavior.csv', FEATENG_DIR + 'remap_dict.pkl', FEATENG_DIR + 'time_distri.png', FEATENG_DIR + 'user_feat_dict.pkl', FEATENG_DIR + 'item_feat_dict.pkl')
    remap(FEATENG_DIR + 'joined_user_behavior.csv', FEATENG_DIR + 'remap_dict.pkl', FEATENG_DIR + 'remap_joined_user_behavior.csv')
    sort_log(FEATENG_DIR + 'remap_joined_user_behavior.csv', FEATENG_DIR + 'sorted_remap_joined_user_behavior.csv')
    gen_target_seq(FEATENG_DIR + 'sorted_remap_joined_user_behavior.csv',
                    FEATENG_DIR + 'target_train.txt', FEATENG_DIR + 'target_vali.txt', FEATENG_DIR + 'target_test.txt', FEATENG_DIR + 'user_seq.txt', FEATENG_DIR + 'database.txt', 
                    FEATENG_DIR + 'context_dict_train.pkl', FEATENG_DIR + 'context_dict_vali.pkl', FEATENG_DIR + 'context_dict_test.pkl')
    insert_elastic(FEATENG_DIR + 'database.txt')
