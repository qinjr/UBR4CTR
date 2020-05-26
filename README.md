# User Behavior Retrieval for CTR Prediction (UBR4CTR)
A `tensorflow` implementation of all the compared models for our SIGIR 2020 paper:

[User Behavior Retrieval for Click-Through Rate Prediction](https://arxiv.org/)

If you have any questions, please contact the author: [Jiarui Qin](http://jiaruiqin.me).


## Abstract
> Click-through rate (CTR) prediction plays a key role in modern online personalization services.
  In practice, it is necessary to capture user's drifting interests by modeling sequential user behaviors to build an accurate CTR prediction model. 
  However, as the users accumulate more and more behavioral data on the platform, it becomes non-trivial for the sequential models to make use of the whole behavior history of each user. First, directly feeding the long behavior sequence will make online inference time and system load infeasible. Second, there is much noise in such long histories to fail the sequential model learning.
  The current industrial solutions mainly truncate the sequences and just feed recent behaviors to the prediction model, which leads to a problem that sequential patterns such as periodicity or long-term dependency are not embedded in the recent several behaviors but in far back history.
  To tackle these issues, in this paper we consider it from the data perspective instead of just designing more sophisticated yet complicated models and propose User Behavior Retrieval for CTR prediction (UBR4CTR) framework. In UBR4CTR, the most relevant and appropriate user behaviors will be firstly retrieved from the entire user history sequence using a learnable search method. These retrieved behaviors are then fed into a deep model to make the final prediction instead of simply using the most recent ones. It is highly feasible to deploy UBR4CTR into industrial model pipeline with low cost. Experiments on three real-world large-scale datasets demonstrate the superiority and efficacy of our proposed framework and models.

## Citation
```
@inproceedings{qin2020user,
	title={User Behavior Retrieval for Click-Through Rate Prediction},
	author={Qin, Jiarui and Zhang, Weinan and Wu, Xin and Jin, Jiarui and Fang, Yuchen and Yu, Yong},
	booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’20)},
	year={2020},
	organization={ACM}
}
```
## Dependencies
- [Tensorflow](https://www.tensorflow.org) >= 1.4
- [Python](https://www.python.org) >= 3.5
- [Elastic Search](https://www.elastic.co)
- [numpy](https://numpy.org)
- [sklearn](https://scikit-learn.org)

## Data Preparation & Preprocessing
- We give a sample raw data in the `data` folder. The full raw datasets are: [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42), [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) and [Alipay](https://tianchi.aliyun.com/dataset/dataDetail?dataId=53). **Remove the first line of table head**.
- Feature Engineering:
```
python3 feateng_tmall.py # for Tmall
python3 feateng_taobao.py # for Taobao
python3 feateng_alipay.py # for Alipay
```


## Train the Models
- To run UBR4CTR, rec_model=['RecAtt', 'RecSum'], ubr_model=['UBR_SA']
```
python3 train.py [rec_model] [ubr_model] [gpu] [dataset]
```

- To run baselines, model_name=['GRU4Rec', 'Caser', 'SASRec', 'HPMN', 'MIMN', 'DIN', 'DIEN']:
```
python3 train_baseline.py [model_name] [gpu] [dataset]
```
