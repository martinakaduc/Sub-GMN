import os
import dgl
import torch
import pickle
import numpy as np
from 数据生成 import *
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset, DataLoader


def onehot_encoding(x, max_x):
    onehot_vector = [0] * max_x
    onehot_vector[x - 1] = 1  # label start from 1
    return onehot_vector


def node_feature(m, node_i, max_nodes):
    node = m.nodes[node_i]
    return onehot_encoding(node["label"], max_nodes)


def onehot_encoding_node(m, embedding_dim):
    H = []
    for i in m.nodes:
        H.append(node_feature(m, i, embedding_dim))
    H = np.array(H)
    return H


class dgraph_v2(Dataset):
    def __init__(self, root_dir, key_file, embedding_dim):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, key_file), "rb") as fp:
            keys = pickle.load(fp)

        self.graph_pairs = list(filter(lambda x: "iso" in x, keys))
        self.embedding_dim = embedding_dim

    def __getitem__(self, index):
        graph_pair_index = self.graph_pairs[index]
        graph_pair_path = os.path.join(self.root_dir, graph_pair_index)

        with open(graph_pair_path, "rb") as f:
            data = pickle.load(f)
            graph_q, graph_da, mapping = data

        # Prepare subgraph
        n1 = graph_q.number_of_nodes()
        H1 = onehot_encoding_node(graph_q, self.embedding_dim)

        # Prepare source graph
        n2 = graph_da.number_of_nodes()
        H2 = onehot_encoding_node(graph_da, self.embedding_dim)

        graph_q = dgl.from_networkx(graph_q)
        graph_da = dgl.from_networkx(graph_da)

        graph_q.ndata['x'] = torch.tensor(H1, dtype=torch.float32)
        graph_da.ndata['x'] = torch.tensor(H2, dtype=torch.float32)

        label = np.zeros((n1, n2))
        if len(mapping) > 0:
            mapping = np.array(mapping).T
            label[mapping[0], mapping[1]] = 1.0

        same_nlb = np.array(same(H1, H2))
        return graph_da, graph_q, label, same_nlb

    def __len__(self):
        return len(self.graph_pairs)


class dgraph(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graph_pairs = os.listdir(self.root_dir)

    def __getitem__(self, index):
        graph_pair_index = self.graph_pairs[index]
        graph_pair_path = os.path.join(self.root_dir, graph_pair_index)
        graph_pair, label_dict = load_graphs(graph_pair_path)
        graph_da = graph_pair[0]
        graph_q = graph_pair[1]
        label = np.array(label_dict['glabel'])
        same = np.array(label_dict['same_m'])
        return graph_da, graph_q, label, same

    def __len__(self):
        return len(self.graph_pairs)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    g1, g2, labels, sames = map(list, zip(*samples))
    bg1 = dgl.batch(g1)
    bg2 = dgl.batch(g2)

    labels_out = torch.zeros((1, bg2.number_of_nodes(), bg1.number_of_nodes()))
    sames_out = torch.zeros((1, bg2.number_of_nodes(), bg1.number_of_nodes()))

    curr_n1 = 0
    curr_n2 = 0
    for l, s in zip(labels, sames):
        labels_out[0, curr_n1:curr_n1 + l.shape[0],
                   curr_n2:curr_n2 + l.shape[1]] = torch.tensor(l)
        sames_out[0, curr_n1:curr_n1 + s.shape[0],
                  curr_n2:curr_n2 + s.shape[1]] = torch.tensor(s)
        curr_n1 += l.shape[0]
        curr_n2 += l.shape[1]

    return bg1, bg2, labels_out, sames_out


if __name__ == "__main__":
    dset = dgraph(root_dir='./数据/train/')
    data_loader = DataLoader(dset, batch_size=8,
                             shuffle=True, collate_fn=collate)
