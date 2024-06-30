
import torch.nn as nn
import torch
from layers import three_gcn, att_layer, NTN


class sub_GMN(torch.nn.Module):
    def __init__(self, GCN_in_size, GCN_out_size, NTN_k, mask=False):
        super(sub_GMN, self).__init__()
        self.GCN_in_size = GCN_in_size
        self.GCN_out_size = GCN_out_size  # D
        self.NTN_k = NTN_k
        self.mask = mask
        # layers
        self.GCN = three_gcn(in_size=self.GCN_in_size,
                             out_size=self.GCN_out_size)
        self.NTN1 = NTN(D=self.GCN_out_size, k=self.NTN_k)
        self.NTN2 = NTN(D=self.GCN_out_size, k=self.NTN_k)
        self.NTN3 = NTN(D=self.GCN_out_size, k=self.NTN_k)
        self.Con1 = nn.Conv2d(self.NTN_k + 1, 1, [1, 1])
        self.Con2 = nn.Conv2d(self.NTN_k + 1, 1, [1, 1])
        self.Con3 = nn.Conv2d(self.NTN_k + 1, 1, [1, 1])
        self.con_end = nn.Conv2d(self.NTN_k * 3 + 6, 1, [1, 1])

    def forward(self, bg_da, bg_q, b_same):  # b_same bx5x8
        b_same_adddim = torch.unsqueeze(b_same, 1)  # b_same bx1x5x8
        da1, da2, da3 = self.GCN(bg_da)
        q1, q2, q3 = self.GCN(bg_q)

        # 1
        att1 = att_layer(batch_q_em=q1, batch_da_em=da1)  # att bx1x5x18
        N1_16 = self.NTN1(batch_q_em=q1, batch_da_em=da1)  # N1_16 bxkx5x18
        N1_16 = N1_16 * att1  # N1_16 bxkx5x18
        he_1 = torch.cat([b_same_adddim, N1_16], dim=1)  # he_1 bx(k+1)x5x18
        end1 = self.Con1(he_1)  # end1 bx1x5x18
        end1 = torch.softmax(end1, dim=3)
        # 2
        att2 = att_layer(batch_q_em=q2, batch_da_em=da2)  # att bx1x5x18
        N2_16 = self.NTN2(batch_q_em=q2, batch_da_em=da2)  # N_16 bxkx5x18
        N2_16 = N2_16 * att2  # N_16 bxkx5x18
        he_2 = torch.cat([b_same_adddim, N2_16], dim=1)  # he bx(k+1)x5x18
        end2 = self.Con2(he_2)  # end bx1x5x18
        end2 = torch.softmax(end2, dim=3)
        # 3
        att3 = att_layer(batch_q_em=q3, batch_da_em=da3)  # att bx1x5x18
        N3_16 = self.NTN3(batch_q_em=q3, batch_da_em=da3)  # N_16 bxkx5x18
        N3_16 = N3_16 * att3  # N_16 bxkx5x18
        he_3 = torch.cat([b_same_adddim, N3_16], dim=1)  # he bx(k+1)x5x18
        end3 = self.Con3(he_3)  # end bx1x5x18
        end3 = torch.softmax(end3, dim=3)
        # end
        end = torch.cat([end1, end2, end3, he_1, he_2, he_3],
                        dim=1)  # end bx(3k+6)x5x18
        end = self.con_end(end)  # end bx1x5x18
        end = end.squeeze(1)  # end bx5x18

        if self.mask:
            end = torch.softmax(end, dim=2)
            end = end * b_same  # end bx5x18
            # end = torch.softmax(end, dim=2)
        else:
            end = torch.softmax(end, dim=2)
        return end
