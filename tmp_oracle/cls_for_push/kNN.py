import pickle, glob
from argparse import ArgumentParser

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier

from utils import TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH, SAVE_PATH
from utils import QUERY_DATA_PATH_1, QUERY_DATA_PATH_2, QUERY_DIR2
from metric import cal_acc_auc, cal_confusion_matrix, cal_fscore

def read_file(args, file_path):
    data = []
    label = []
    for line in open(file_path, 'r'):
        l, _, seq_or_feat = line.strip().partition(',')
        label.append(int(l))
        if args.feat:
            data.append(np.asarray([float(n) for n in seq_or_feat.split(',')]))
        else:
            data.append(seq_or_feat)
    
    return data, label

def vectorize_seq(args, data):
    # the data is a list of original sequences.
    if args.mode == 0:
        amp_vzr = CountVectorizer(lowercase=False, analyzer='char', ngram_range=(args.n_gram_min, args.n_gram_max))
        amp_counts = amp_vzr.fit_transform(data)
        # create a transformer, to convert raw frequency counts into TF-IDF values
        amp_tfr = TfidfTransformer()
        amp_tfidf = amp_tfr.fit_transform(amp_counts)
        save_file = SAVE_PATH + 'knn-ngram_vzr:{0}-{1}-{2}-{3}-{4}'.format(\
            args.n_neighbors, args.n_jobs, args.n_gram_min, args.n_gram_max, args.weights) + '.pkl'
        with open(save_file, 'wb') as fout:
            pickle.dump((amp_vzr, amp_tfr), fout)
    else:
        load_file = SAVE_PATH + 'knn-ngram_vzr:{0}-{1}-{2}-{3}-{4}'.format(\
            args.n_neighbors, args.n_jobs, args.n_gram_min, args.n_gram_max, args.weights) + '.pkl'
        print("Load {0} model from {1} successfully!".format('ngram_vzr', load_file))
        with open(load_file, 'rb') as f:
            amp_vzr, amp_tfr = pickle.load(f)
            amp_counts = amp_vzr.transform(data)
            amp_tfidf = amp_tfr.transform(amp_counts)

    return amp_tfidf

def train(args):
    # load training data
    train_x, train_y = read_file(args, TRAIN_DATA_PATH)
    # create a KNearstNeighbor classifier
    kNN = KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                n_jobs=args.n_jobs,
                                weights=args.weights, # weight function to value each points
                                )
    if args.feat == 0:
        train_x = vectorize_seq(args, train_x)
    kNN.fit(train_x, train_y)

    save_file = SAVE_PATH + 'kNN-model:{0}-{1}-{2}-{3}-{4}'.format(\
                args.n_neighbors, args.n_jobs, args.n_gram_min, args.n_gram_max, args.weights) + '.pkl'
    with open(save_file, 'wb') as fout:
        pickle.dump(kNN, fout)

def test(args, data_path, **kwargs):
    # load test data
    
    test_x, test_y = read_file(args, data_path)
    if args.feat == 0:
        test_x = vectorize_seq(args, test_x)
    load_file = SAVE_PATH + 'kNN-model:{0}-{1}-{2}-{3}-{4}'.format(\
            args.n_neighbors, args.n_jobs, args.n_gram_min, args.n_gram_max, args.weights) + '.pkl'
    print("Load {0} model from {1} successfully!".format('KNN', load_file))
    with open(load_file, 'rb') as f:
        kNN = pickle.load(f)
    pred_label = kNN.predict(test_x)
    pred_proba = kNN.predict_proba(test_x)
    
    
    if args.mode == 2: # test mode, calculate some statistics
        acc, auc = cal_acc_auc(test_y, pred_label, pred_proba)
        tn, fp, fn, tp = cal_confusion_matrix(test_y, pred_label)
        prc, rcl, f1 = cal_fscore(test_y, pred_label)
        
        print("[eval_kNN]==> eval_acc:{:.4f} eval_auc:{:.4f}"\
                        "tn:{:.4f} fp:{:.4f} fn:{:.4f} tp:{:.4f}"\
                        "prc:{:.4f} rcl:{:.4f} f1:{:.4f}"\
                        .format(acc, auc, tn, fp, fn, tp, prc, rcl, f1))
        return acc, auc
    elif args.mode == 2: # query mode
        np.savetxt('proba_{0}_{1}'.format('kNN', data_path.split('/')[-1]), pred_proba, delimiter=',')
    else:
        raise NotImplementedError


def run_cli():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0:train, 1:test, 2:query')
    parser.add_argument('--feat', type=int, default=0)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=10)
    parser.add_argument('--weights', type=str, default='uniform')
    parser.add_argument('--n_gram_min', type=int, default=1)
    parser.add_argument('--n_gram_max', type=int, default=2)

    args = parser.parse_args()
    if args.mode == 0: # training
        train(args)
    elif args.mode == 1: # test a single file
        test(args, TEST_DATA_PATH)
    elif args.mode == 2: # query a few files under a directory
        for data_path in glob.glob(QUERY_DIR2 + '/*.csv'):
            test(args, data_path)
    else:
        raise NotImplementedError
        
if __name__ == '__main__':
    run_cli()
