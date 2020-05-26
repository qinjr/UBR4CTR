import sys
import pickle as pkl
import time
import numpy as np
import random
from common import *
from elastic_client import *
import multiprocessing

random.seed(1111)

class DataLoader(object):
    def __init__(self, 
                batch_size, 
                seq_file,
                target_file,
                user_feat_dict_file,
                item_feat_dict_file,
                max_len):
        self.batch_size = batch_size
        self.seq_file = open(seq_file, 'r')
        self.target_file = open(target_file, 'r')

        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        else:
            self.user_feat_dict = None
        
        # item has to have multiple feature fields
        with open(item_feat_dict_file, 'rb') as f:
            self.item_feat_dict = pkl.load(f)

        self.max_len = max_len

    def __iter__(self):
        return self
    
    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        label_batch = []
        user_seq_batch = []
        user_seq_len_batch = []

        for i in range(self.batch_size):
            seq_line = self.seq_file.readline()
            target_line = self.target_file.readline()
            if seq_line == '':
                raise StopIteration

            target_uid, target_iid = target_line[:-1].split(',')
            if self.user_feat_dict != None:
                target_user_batch.append([int(target_uid)] + self.user_feat_dict[target_uid])
            else:
                target_user_batch.append([int(target_uid)])

            target_item_batch.append([int(target_iid)] + self.item_feat_dict[target_iid])
            if i % 2 == 0:
                label_batch.append(1)
            else:
                label_batch.append(0)
            
            seq = seq_line[:-1].split(',')
            seqlen = len(seq)
            user_seq = []
            for iid in seq:
                item = [int(iid)] + self.item_feat_dict[iid]
                user_seq.append(item)
            if seqlen >= self.max_len:
                user_seq = user_seq[-self.max_len:]
                user_seq_len_batch.append(self.max_len)
            else:
                user_seq += [[0] * len(user_seq[-1])] * (self.max_len - seqlen)
                user_seq_len_batch.append(seqlen)
            user_seq_batch.append(user_seq)

        
        return [user_seq_batch, user_seq_len_batch, target_user_batch, target_item_batch, label_batch]

class DataLoader_Target(object):
    def __init__(self, 
                batch_size, 
                target_file,
                user_feat_dict_file,
                item_feat_dict_file,
                context_dict_file):
        self.batch_size = batch_size
        self.target_file = open(target_file, 'r')

        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        else:
            self.user_feat_dict = None

        with open(item_feat_dict_file, 'rb') as f:
            self.item_feat_dict = pkl.load(f)
        with open(context_dict_file, 'rb') as f:
            self.context_dict = pkl.load(f)

    def __iter__(self):
        return self
    
    def __next__(self):
        target_batch = []
        label_batch = []

        for i in range(self.batch_size):
            target_line = self.target_file.readline()
            if target_line == '':
                raise StopIteration

            target_uid, target_iid = target_line[:-1].split(',')
            if self.user_feat_dict != None:
                target_batch.append([int(target_uid)] + self.user_feat_dict[target_uid] + [int(target_iid)] + self.item_feat_dict[target_iid] + self.context_dict[target_uid])
            else:
                target_batch.append([int(target_uid)] + [int(target_iid)] + self.item_feat_dict[target_iid] + self.context_dict[target_uid])

            if i % 2 == 0:
                label_batch.append(1)
            else:
                label_batch.append(0)
        return target_batch, label_batch

class Taker(object):
    def __init__(self, es_reader, batch_size, b_num, record_fnum):
        self.es_reader = es_reader
        self.batch_size = batch_size
        self.b_num = b_num
        self.record_fnum = record_fnum
    
    def take_behave(self, target_batch, index_batch):
        seq_batch = []
        seq_len_batch = [self.b_num] * self.batch_size

        queries = []
        for i in range(self.batch_size):
            target = np.array(target_batch[i][1:]) # with out uid
            index = np.array(index_batch[i]) # F-1
            query_tup = (str(target_batch[i][0]), ','.join(list(map(str, target[index==1].tolist()))))
            queries.append(query_tup)
        seq_batch = self.es_reader.query(queries, self.b_num, self.record_fnum)

        return seq_batch, seq_len_batch


class DataLoader_Multi(object):
    def __init__(self, workload_list, taker, worker_num=2, wait_time=0.001):
        self.taker = taker
        self.worker_num = worker_num
        self.wait_time = wait_time
        self.threads = []
        self.work = multiprocessing.Queue()
        self.res = multiprocessing.Queue()
        
        for workload_tuple in workload_list:
            self.work.put(workload_tuple)
        print("workload queue size: {}".format(self.work.qsize()))

        for i in range(self.worker_num):
            thread = multiprocessing.Process(target=self.worker)
            self.threads.append(thread)
            thread.daemon = True
            thread.start()
        
    def worker(self):
        while self.work.empty() == False:
            target_batch, label_batch, index_batch = self.work.get()
            seq_batch, seq_len_batch = self.taker.take_behave(target_batch, index_batch)
            self.res.put([seq_batch, seq_len_batch, target_batch, label_batch])
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.res.empty():
            if self.work.empty():
                for thread in self.threads:
                    thread.terminate()
                raise StopIteration
            else:
                time.sleep(self.wait_time)
    
        re = self.res.get()
        return re