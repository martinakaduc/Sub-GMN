import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader

from dset import dgraph_v2, collate
from net import sub_GMN
from zzh import Regularization
from utils import to_predict_matching, acc_renzao
from sklearn.metrics import f1_score


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed", type=int, default=42)
    parser.add_argument(
        "--data_path", help="path to the data", type=str, default="data_processed"
    )
    parser.add_argument("--ckpt", help="Load ckpt file", type=str, default="")
    parser.add_argument(
        "--train_keys", help="train keys", type=str, default="train_keys.pkl"
    )
    parser.add_argument(
        "--test_keys", help="test keys", type=str, default="test_keys.pkl"
    )
    parser.add_argument("--lr", help="learning rate",
                        type=float, default=1e-3)
    parser.add_argument("--wd", help="weight decay",
                        type=float, default=1e-2)
    parser.add_argument(
        "--d_graph_layer", help="dimension of GNN layer", type=int, default=128
    )
    parser.add_argument("--epoch", help="epoch", type=int, default=30)
    parser.add_argument("--batch_size", help="batch_size",
                        type=int, default=32)
    parser.add_argument(
        "--embedding_dim",
        help="node embedding dim aka number of distinct node label",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    GCN_in_size = args.embedding_dim
    GCN_out_size = args.d_graph_layer
    NTN_k = 16

    d_test = dgraph_v2(root_dir=args.data_path,
                       key_file=args.test_keys, embedding_dim=args.embedding_dim)
    dset = dgraph_v2(root_dir=args.data_path,
                     key_file=args.train_keys, embedding_dim=args.embedding_dim)
    batch_size = args.batch_size
    epochs = args.epoch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight_decay = args.wd
    reg_ture = True

    data_loader = DataLoader(dset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate)
    data_test = DataLoader(d_test, batch_size=batch_size,
                           shuffle=False, collate_fn=collate)

    model = sub_GMN(GCN_in_size, GCN_out_size, NTN_k, mask=True)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt))
    model.train()
    model.to(device)

    reg = Regularization(model, weight_decay, p=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss().to(device)

    min_test_loss = 10
    max_acc = 0
    f1_max = 0

    for i in np.arange(epochs):
        print('epochs:  ', i)
        epochs_loss = []
        for j, (bbg1, bbg2, lllabel, same) in enumerate(data_loader):
            # print('batch:  ', j)
            bbg1 = bbg1.to(device)
            bbg2 = bbg2.to(device)
            b_lllabel = lllabel.to(device)
            b_same = same.to(device)
            y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
            y_Hat = torch.masked_select(y_hat, torch.tensor(
                b_same, dtype=torch.bool).to(device))
            y_label = torch.masked_select(b_lllabel, torch.tensor(
                b_same, dtype=torch.bool).to(device))
            # print(y_Hat)
            # print(y_label)
            loss = criterion(y_Hat, y_label)

            # loss = criterion(y_hat, b_lllabel)
            if reg_ture:
                loss = loss + reg(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochs_loss.append(float(loss.detach()))
            # print(loss.detach().cpu().numpy())
            # print('batch loss:  ', float(loss.detach()))
        print('!!!!!!!!epochs_loss:  ', np.mean(epochs_loss))
        torch.save(model.state_dict(), './train.pkl')

        for k, (bbg1, bbg2, lllabel, same) in enumerate(data_test):
            print('test!!!!!!:  ', k)
            bbg1 = bbg1.to(device)
            bbg2 = bbg2.to(device)
            b_lllabel = lllabel.to(device)
            b_same = same.to(device)
            with torch.no_grad():
                y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
                pass
            # y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
            loss = criterion(y_hat, b_lllabel)
            print('min_test_loss:  ', min_test_loss)
            print('test_loss!!!!:  ', np.float(loss.detach()))
            pre_matching = to_predict_matching(y_hat.detach().cpu().numpy())
            print(same.numpy()[3])
            print(y_hat.detach().cpu().numpy()[3])
            print(pre_matching[3])
            acc = acc_renzao(pre_matching, q_size, da_size)
            f1 = f1_score(y_true=list(b_lllabel.cpu().numpy().reshape(-1)), y_pred=list(pre_matching.reshape(-1)),
                          average='binary', pos_label=1)
            print('max_acc      :  ', max_acc)
            print('acc          :  ', acc)
            print('f1_max       :  ', f1_max)
            print('f1           :  ', f1)

            if acc > max_acc:
                torch.save(model.state_dict(), './max_acc.pkl')
                max_acc = acc
            if np.float(loss.detach()) < min_test_loss:
                torch.save(model.state_dict(), './min_loss.pkl')
                min_test_loss = np.float(loss.detach())
            if f1 > f1_max:
                torch.save(model.state_dict(), './max_f1.pkl')
                f1_max = f1


if __name__ == "__main__":
    args = parse_args()
    main(args)
