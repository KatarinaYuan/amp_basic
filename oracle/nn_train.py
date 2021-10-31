import argparse
from oracle.dataset import MAX_SEQ_LENGTH, VOCAB_SIZE
from tqdm import tqdm 
import sys
import ipdb
import os 

import numpy as np

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from oracle.nn_model import LstmCLS, TransformerCLS
from oracle.metric import cal_acc_auc, cal_confusion_matrix, cal_fscore

SAVE_PATH = '/home2/yxy/tmp/amp_basic/run_oracle_train/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-type', type=str, default='input_ids', choices=['input_ids', 'feat_embed']) # NOTE
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001) # 0.05 for Transformer, 0.001 for LSTM 
    parser.add_argument('--clip', type=float, default=1.0)

    # Model
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--output-size', type=int, default=1)
    parser.add_argument('--num-layers', type=int, default=1) # 2 for LSTM 

    args = parser.parse_known_args()[0]

    return args 

def build_model(args):
    model = None 
    if args.oracle_type == 'LSTM':
        model = LstmCLS(vocab_size=VOCAB_SIZE,
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        output_size=args.output_size,
                        num_layers=args.num_layers,
                        max_len=MAX_SEQ_LENGTH,
                        dropout=args.dropout).to(args.device)
    elif args.oracle_type == 'Transformer':
        model = TransformerCLS(vocab_size=VOCAB_SIZE,
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        output_size=args.output_size,
                        num_layers=args.num_layers,
                        max_len=MAX_SEQ_LENGTH,
                        dropout=args.dropout,
                        nhead=args.nhead).to(args.device)
    else:
        raise NotImplementedError
    
    return model 

def evaluate(args, model, eval_dataset, mode='eval'):
    data_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    loss, prob, pred, y_true = [], [], [], []

    with torch.no_grad():

        for idx, batch_x in enumerate(data_loader):
            # forward and get prediction & loss
            logits = model(batch_x['input_ids'].to(args.device), batch_x['attention_mask'].to(args.device))
            b_prob, b_pred = model.predict(logits)
            batch_loss = model.compute_loss(logits.squeeze(), batch_x['label'].to(args.device))

            loss.append(batch_loss.item())
            prob.append(b_prob)
            pred.append(b_pred)
            y_true.append(batch_x['label'].cpu().numpy())

    loss = np.mean(loss)
    prob = np.concatenate(prob, axis=0)
    pred = np.concatenate(pred, axis=0) # NOTE
    y_true = np.concatenate(y_true, axis=0)

    acc, auc = cal_acc_auc(y_true, pred, prob)
    tn, fp, fn, tp = cal_confusion_matrix(y_true, pred)
    prc, rcl, f1 = cal_fscore(y_true, pred)

    print("==> [{}_result]  total_data:{} ---\n"\
                "{}_loss:{:.4f} {}_acc:{:.4f} {}_auc:{:.4f};\n"\
                "tn:{:.4f} fp:{:.4f} fn:{:.4f} tp:{:.4f};\n"\
                "prc:{:.4f} rcl:{:.4f} f1:{:.4f};\n"\
                .format(mode, len(eval_dataset), 
                    mode, loss, mode, acc, mode, auc, tn, fp, fn, tp, prc, rcl, f1))

    return loss, acc, auc, prob, pred


def train(args, model, preprocessed_datasets):

    print("==> Begin training, Num of Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    data_loader = DataLoader(preprocessed_datasets['train'], batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_eval = (0., 0., 0.)

    #ipdb.set_trace()

    for epoch in tqdm(range(args.epochs)):
        # start training loop
        model.train()
        for idx, batch_x in enumerate(data_loader):

            # forward and get prediction & loss
            logits = model(batch_x['input_ids'].to(args.device), batch_x['attention_mask'].to(args.device))
            b_prob, b_pred = model.predict(logits)
            batch_loss = model.compute_loss(logits.squeeze(), batch_x['label'].to(args.device))
            #ipdb.set_trace()
            # update
            optimizer.zero_grad()
            batch_loss.backward()
            clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if idx % args.verbo == 0:
                print('[Epoch:{} Step:{} loss:{:.4f}]'.format(epoch, idx, batch_loss.item()))
                sys.stdout.flush()

        eval_loss, eval_acc, eval_auc, _, _ = evaluate(args, model, preprocessed_datasets['validation'])
        
        if eval_auc > best_eval[2]:
            best_eval = (eval_loss, eval_acc, eval_auc)
            save_model(model, os.path.join(SAVE_PATH, args.feature_type, 'nn', f'{args.oracle_type}.ckpt'))

def test(args, test_dataset):
    #assert 0 
    load_file = os.path.join(SAVE_PATH, args.feature_type, 'nn', f'{args.oracle_type}.ckpt')
    model = build_model(args)
    model = load_model(model, load_file)

    #ipdb.set_trace()
    loss, acc, auc, prob, pred = evaluate(args, model, test_dataset, 'test')

    return acc, auc, prob, pred

def save_model(model, save_path):
    print("==> Save model to {}".format(save_path))
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    print("==> Load model from {}".format(save_path))
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    return model
