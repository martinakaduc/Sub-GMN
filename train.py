import os
import torch
import time
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from dset import dgraph_v2, collate
from net import sub_GMN
from zzh import Regularization
from utils import to_predict_matching, acc_renzao, eval_mapping
from sklearn.metrics import f1_score


def set_seed(seed):
    random.seed(seed)
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

    result_dir = os.path.join("results/", args.data_path.split("/")[-1])
    ckpt_dir = os.path.join("ckpts", args.data_path.split("/")[-1])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    result_file = f"result_matching{args.test_keys[9:-4]}.csv"

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

    early_stop = 0
    max_mrr = 0
    # max_acc = 0
    # f1_max = 0

    for i in np.arange(epochs):
        print('epochs:  ', i)
        epochs_loss = []
        for j, (bbg1, bbg2, lllabel, same) in enumerate(tqdm(data_loader, desc=f"Training Epoch {i}")):
            bbg1 = bbg1.to(device)
            bbg2 = bbg2.to(device)
            b_lllabel = lllabel.to(device)
            b_same = same.to(device)
            y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
            y_Hat = torch.masked_select(y_hat, torch.tensor(
                b_same, dtype=torch.bool).to(device))
            y_label = torch.masked_select(b_lllabel, torch.tensor(
                b_same, dtype=torch.bool).to(device))

            loss = criterion(y_Hat, y_label)

            if reg_ture:
                loss = loss + reg(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochs_loss.append(float(loss.detach()))

        print('!!!!!!!!epochs_loss:  ', np.mean(epochs_loss))

        list_results = []
        start = time.time()
        for k, (bbg1, bbg2, lllabel, same) in enumerate(tqdm(data_test, desc=f"Testing Epoch {i}")):
            bbg1 = bbg1.to(device)
            bbg2 = bbg2.to(device)
            b_same = same.to(device)
            with torch.no_grad():
                y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)

            # y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
            # loss = criterion(y_hat, b_lllabel)
            # pre_matching = to_predict_matching(y_hat.detach().cpu().numpy())
            # acc = acc_renzao(
            #     pre_matching, bbg2.number_of_nodes(), bbg1.number_of_nodes())
            # f1 = f1_score(y_true=list(b_lllabel.cpu().numpy().reshape(-1)), y_pred=list(pre_matching.reshape(-1)),
            #               average='binary', pos_label=1)

            gt_mapping = {}
            x_coord, y_coord = np.where(lllabel[0] > 0)
            for x, y in zip(x_coord, y_coord):
                gt_mapping[x] = [y]  # Subgraph node: Graph node

            pred_mapping = defaultdict(lambda: {})
            mapping_pred = y_hat[0].detach().cpu().numpy()
            x_coord, y_coord = np.where(mapping_pred > 0)

            for x, y in zip(x_coord, y_coord):
                pred_mapping[x][y] = mapping_pred[
                    x, y
                ]  # Subgraph node: Graph node

            sorted_predict_mapping = defaultdict(lambda: [])
            sorted_predict_mapping.update(
                {
                    k: [
                        y[0]
                        for y in sorted(
                            [(n, prob) for n, prob in v.items()],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                    for k, v in pred_mapping.items()
                }
            )

            eval_mapping_results = eval_mapping(
                gt_mapping, sorted_predict_mapping)

            list_results.append(eval_mapping_results)

        end = time.time()

        list_results = np.array(list_results)
        avg_results = np.mean(list_results, axis=0)

        print("Test time: ", end - start)
        print("Top1-Top10 Accuracy, MRR")
        print(avg_results)

        if avg_results[-1] > max_mrr:
            early_stop = 0
            max_mrr = avg_results[-1]
            torch.save(model.state_dict(), os.path.join(
                ckpt_dir, f"best_model.pkl"))

            with open(
                os.path.join(result_dir, result_file),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    "Time,Top1-Acc,Top2-Acc,Top3-Acc,Top4-Acc,Top5-Acc,Top6-Acc,Top7-Acc,Top8-Acc,Top9-Acc,Top10-Acc,MRR\n"
                )
                f.write("%f," % (end - start))
                f.write(",".join([str(x) for x in avg_results]))
                f.write("\n")
        else:
            early_stop += 1

        if early_stop > 3:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
