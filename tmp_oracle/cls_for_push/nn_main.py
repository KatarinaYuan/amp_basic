import sys
import torch
import numpy as np
import glob

from argparse import ArgumentParser
from tqdm import tqdm 

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from utils import get_data4oracle, PADDING_LEN, VOCAB_SIZE, MAX_LEN
from utils import TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH, SAVE_PATH, QUERY_DIR2
from nn_model import LSTM_CLS, TransformerCLS
from metric import cal_acc_auc, cal_confusion_matrix, cal_fscore


def run_cli():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', type=int, default=0, help='0:train, 1:test, 2:query')
    parser.add_argument('--nhead', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='lstm')

    parser.add_argument('--verbo', type=int, default=10)
    args = parser.parse_args()

    model = None
    if args.model == 'lstm':
        model = LSTM_CLS(vocab_size=VOCAB_SIZE,
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        output_size=args.output_size,
                        num_layers=args.num_layers,
                        max_len=MAX_LEN,
                        dropout=args.dropout,
                        pos_weight=None).cuda()
    elif args.model == 'tfr':
        model = TransformerCLS(vocab_size=VOCAB_SIZE,
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        output_size=args.output_size,
                        num_layers=args.num_layers,
                        max_len=MAX_LEN,
                        dropout=args.dropout,
                        nhead=args.nhead,
                        pos_weight=None).cuda()
    else:
        raise NotImplementedError
    
    if args.mode == 0: # training
        train(args, model)
    elif args.mode == 1: # testing
        test(args, model, TEST_DATA_PATH)
    elif args.mode == 2: # query a few files under a directory
        for data_path in glob.glob(QUERY_DIR2 + '/*.csv'):
            test(args, model, data_path)
    else:
        raise NotImplementedError

def evaluate(model, X_ids_val, y_val, args):
    model.eval()

    loss = []
    prob, pred = [], []

    with torch.no_grad():
        for idx in range(0, X_ids_val.shape[0], args.batch_size):
            batch_x = X_ids_val[idx: idx+args.batch_size]
            batch_y = y_val[idx: idx+args.batch_size]

            logits = model(batch_x)
            b_prob, b_pred = model.predict(logits)
            batch_loss = model.compute_loss(logits.squeeze(), batch_y)

            loss.append(batch_loss.item())
            prob.append(b_prob)
            pred.append(b_pred)

    loss = np.mean(loss)
    prob = np.concatenate(prob, axis=0)
    pred = np.concatenate(pred, axis=0)
    
    y_true = y_val.detach().cpu().numpy()
    acc, auc = cal_acc_auc(y_true, pred, prob)
    tn, fp, fn, tp = cal_confusion_matrix(y_true, pred)
    prc, rcl, f1 = cal_fscore(y_true, pred)

    print("[eval_result]==> eval_loss:{:.4f} eval_acc:{:.4f} eval_auc:{:.4f}"\
                    "tn:{:.4f} fp:{:.4f} fn:{:.4f} tp:{:.4f}"\
                    "prc:{:.4f} rcl:{:.4f} f1:{:.4f}"\
                    .format(loss, acc, auc, tn, fp, fn, tp, prc, rcl, f1))

    return loss, acc, auc, prob, pred

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    return model

def test(args, model, data_path):
    X_ids_test, y_test = get_data4oracle(data_path)
    X_ids_test, y_test = X_ids_test.cuda(), y_test.cuda()

    model = load_model(model, save_path=SAVE_PATH + args.model)
    loss, cnt = 0., 0
    best_eval = (0., 0., 0.)
    
    # evaluation
    eval_loss, eval_acc, eval_auc, prob, pred = evaluate(model, X_ids_test, y_test, args)
    if args.mode == 1: # testing, print acc and auc only.
        print("Acc: {}".format(eval_acc))
        print("Auc: {}".format(eval_auc))
    elif args.mode == 2: # query a file consisting generated sequences.
        np.savetxt('proba_{0}_{1}'.format(args.model, data_path.split('/')[-1]), prob, delimiter=',')
    else:
        raise NotImplementedError
        

def train(args, model):
    X_ids_train, y_train = get_data4oracle(TRAIN_DATA_PATH)
    X_ids_val, y_val = get_data4oracle(VAL_DATA_PATH)
    X_ids_train, y_train = X_ids_train.cuda(), y_train.cuda()
    X_ids_val, y_val = X_ids_val.cuda(), y_val.cuda()
    
    optimizer = Adam(model.parameters(),
                    lr=args.lr)

    if args.weight > 0.:
        weight = torch.full((y_train.shape[0],), args.weight, dtype=torch.float32).cuda()
        weight[y_train == 0] = 1.0
        data_loader = DataLoader(list(zip(X_ids_train, y_train)), batch_size=args.batch_size, 
                    sampler=WeightedRandomSampler(weight, num_samples=y_train.shape[0]))
    else:
        data_loader = DataLoader(list(zip(X_ids_train, y_train)), batch_size=args.batch_size, 
                            shuffle=True)
    
    loss, cnt = 0., 0
    best_eval = (0., 0., 0.)
    
    for epoch in tqdm(range(args.epochs)):
        # start training loop
        model.train()
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            # forward and get prediction & loss
            logits = model(batch_x)
            b_prob, b_pred = model.predict(logits)
            batch_loss = model.compute_loss(logits.squeeze(), batch_y)

            # update
            optimizer.zero_grad()
            batch_loss.backward()
            clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if idx % args.verbo == 0:
                print('[Epoch:{} Step:{} loss:{:.4f}]'.format(epoch, idx, batch_loss.item()))
                sys.stdout.flush()
        eval_loss, eval_acc, eval_auc, _, _ = evaluate(model, X_ids_val, y_val, args)

        if eval_auc > best_eval[2]:
            best_eval = (eval_loss, eval_acc, eval_auc)
            print("==> save model")
            save_model(model, SAVE_PATH + args.model)

if __name__ == '__main__':
    run_cli()
