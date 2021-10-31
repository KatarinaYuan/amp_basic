import os 
import pickle
import ipdb
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from oracle.metric import cal_acc_auc, cal_confusion_matrix, cal_fscore

SAVE_PATH = '/home2/yxy/tmp/amp_basic/run_oracle_train/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-jobs', type=int, default=10)
    # KNN
    parser.add_argument('--n-neighbors', type=int, default=3)
    parser.add_argument('--weights', type=str, default='uniform')
    # RandomForest
    parser.add_argument('--n_trees', type=int, default=160)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--pos_weight', type=float, default=1.)

    args = parser.parse_known_args()[0]

    return args 

def build_model(args):
    if args.oracle_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                    n_jobs=args.n_jobs,
                                    weights=args.weights, # weight function to value each points
                                    )
        args.oracle_detail = "KNN-model:{}_{}_{}".format(
                            args.n_neighbors, args.n_jobs, args.weights)
    elif args.oracle_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=args.n_trees,
                                max_depth=args.depth,
                                n_jobs=args.n_jobs,
                                class_weight={0:1, 1:args.pos_weight}, # control the weight of positive samples
                                )
        args.oracle_detail = "RandomForest-model:{}_{}_{}_{:.2f}".format(
                            args.n_trees, args.depth, args.n_jobs, args.pos_weight)
    else:
        raise NotImplementedError
    return model 

def evaluate(args, model, eval_dataset, mode='eval'):

    test_x = eval_dataset["feat_"+args.feature_type]
    test_y = eval_dataset['label']

    pred_label = model.predict(test_x)
    pred_proba = model.predict_proba(test_x)

    acc, auc = cal_acc_auc(test_y, pred_label, pred_proba)
    tn, fp, fn, tp = cal_confusion_matrix(test_y, pred_label)
    prc, rcl, f1 = cal_fscore(test_y, pred_label)
    
    print("==> [{}_result]  total_data:{} feature_type:{} ---\n"\
                "{}_acc:{:.4f} {}_auc:{:.4f};\n"\
                "tn:{:.4f} fp:{:.4f} fn:{:.4f} tp:{:.4f};\n"\
                "prc:{:.4f} rcl:{:.4f} f1:{:.4f};\n"\
                .format(mode, len(eval_dataset), args.feature_type,
                    mode, acc, mode, auc, tn, fp, fn, tp, prc, rcl, f1))

    
    return acc, auc, pred_proba, pred_label


def train(args, model, preprocessed_datasets):

    # Train
    train_x = preprocessed_datasets['train']["feat_"+args.feature_type]
    train_y = preprocessed_datasets['train']['label']
    model.fit(train_x, train_y)

    # Evaluate
    acc, auc = evaluate(args, model, preprocessed_datasets['validation'])

    save_file = os.path.join(SAVE_PATH, args.feature_type, 'sklearn',  f'{args.oracle_type}.pkl')
    #save_file = os.path.join(args.load_dir, args.oracle_detail + '-' + args.feature_detail + '.pkl')
    save_model(model, save_file)

def test(args, test_dataset):

    #load_file = os.path.join(args.load_dir, args.oracle_detail + '-' + args.feature_detail + '.pkl')
    load_file = os.path.join(SAVE_PATH, args.feature_type, 'sklearn', f'{args.oracle_type}.pkl')
    model = load_model(load_file)
    
    acc, auc, prob, pred = evaluate(args, model, test_dataset, 'test')

    return acc, auc, prob, pred

def save_model(model, save_path):
    print("==> Save model to {}".format(save_path))
    with open(save_path, 'wb') as fout:
        pickle.dump(model, fout)

def load_model(save_path):
    print("==> Load model from {}".format(save_path))
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    return model 