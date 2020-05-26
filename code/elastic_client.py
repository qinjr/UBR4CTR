from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
import elasticsearch.helpers
import time
import numpy as np

class ESWriter(object):
    def __init__(self, input_file, index_name, host_url = 'localhost:9200'):
        self.input_file = input_file
        self.es = Elasticsearch(host_url)
        self.index_name = index_name
        
        # delete if there is old
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])

        self.create_index_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "my_analyzer": {
                        "tokenizer": "my_tokenizer"
                        }
                    },
                    "tokenizer": {
                        "my_tokenizer": {
                        "type": "pattern",
                        "pattern": ","
                        }
                    }
                },
            
            },
            "mappings": {
                "properties": {
                "record": {
                    "type": "text",
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer'
                    }
                }
            },
        }
        self.es.indices.create(index=self.index_name, body=self.create_index_body, ignore=400)
        print('index created')

    def write(self):
        t = time.time()
        with open(self.input_file, 'r') as f:
            docs = []
            batch_num = 0
            for line in f:
                line_item = line[:-1].split(',')
                userid = line_item[0]
                record = ','.join(line_item[1:-1])
                timestamp = int(line_item[-1])
                doc = {
                    'userid' : userid,
                    'record': record,
                    'timestamp': timestamp
                }
                docs.append(doc)
                if len(docs) == 1000:
                    actions = [{
                        '_op_type': 'index',
                        '_index': self.index_name,  
                        '_source': d
                    } 
                    for d in docs]

                    elasticsearch.helpers.bulk(self.es, actions)
                    batch_num += 1
                    docs = []
                    if batch_num % 1000 == 0:
                        print('{} data samples have been inserted'.format(batch_num * 1000))
            
            # the last bulk
            if docs != []:
                actions = [{
                    '_op_type': 'index',
                    '_index': self.index_name,  
                    '_source': d
                } 
                for d in docs]
                elasticsearch.helpers.bulk(self.es, actions)
        
        print('data insert time: %.2f seconds' % (time.time() - t))


class ESReader(object):
    def __init__(self, index_name, host_url = 'localhost:9200'):
        self.es = Elasticsearch(host_url)
        self.index_name = index_name

    def query(self, queries, size, record_fnum):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for q in queries:
            s = Search().query("match", userid=q[0]).query("match", record=q[1])[:size]
            ms = ms.add(s)
        responses = ms.execute()

        res_batch = []
        for response in responses:
            res = []
            for hit in response:
                res.append([int(hit.userid)] + list(map(int, hit.record.split(','))))
            if len(res) < size:
                res += [np.zeros([record_fnum,]).astype(np.int32).tolist()] * (size - len(res))
            res_batch.append(res)
        return res_batch
        