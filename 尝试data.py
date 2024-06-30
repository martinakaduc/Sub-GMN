import torch
from torch.utils.data import DataLoader
from dset import dgraph, collate
from net import sub_GMN

device = torch.device('cuda:0')

GCN_in_size, GCN_out_size, q_size, da_size, NTN_k = 10, 128, 5, 18, 16


md = sub_GMN(GCN_in_size, GCN_out_size, q_size, da_size, NTN_k)
md.train()
md.to(device)


a = './数据/train/'
dset = dgraph(root_dir=a)

data_loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=collate)

# nm = iso.numerical_node_match('x', 1.0)
# aaaa = torch.arange(10, dtype=torch.float32).reshape(10,1)
for i, (bbg1, bbg2, lllabel, same) in enumerate(data_loader):
    print(i)
    bbg1.to(device)
    bbg2.to(device)
    bsame = same.to(device)
    y = md(bg_da=bbg1, bg_q=bbg2, b_same=bsame)
    print(y)

    # print(bbg1)
    # print(bbg2)
    # print(lllabel)
    # print(m)
    # print(lllabel.shape)
    # print(m.shape)
