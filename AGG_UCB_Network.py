import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

# CUDA_VISIBLE_DEVICES=1
class Aggr_module_PR(nn.Module):
    def __init__(self, A, input_dim, embed_dim, PPR_m, dual_layer=False, model_mode_flag=None):
        super(Aggr_module_PR, self).__init__()

        # Param
        self.A = A
        self.PPR_m = PPR_m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.model_mode_flag = model_mode_flag

        #
        self.fc_1 = nn.Linear(input_dim, embed_dim, bias=False)
        self.act = nn.ReLU()

        # Additional linear transformation after ReLU
        self.fc_2 = nn.Linear(embed_dim, embed_dim, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        # First Layer --- SGC
        out = self.fc_1(seq)
        # Out Activation
        out = self.act(out)
        out = self.fc_2(out)

        #
        aggr_c = torch.bmm(adj, out)

        return aggr_c


# =====================================================================================================================
# =====================================================================================================================

# CUDA_VISIBLE_DEVICES=1
class Aggr_module(nn.Module):
    def __init__(self, A, input_dim, embed_dim, dual_layer=False, model_mode_flag=None):
        super(Aggr_module, self).__init__()

        # Param
        self.A = A
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.model_mode_flag = model_mode_flag
        self.dual_layer = dual_layer

        #
        if self.model_mode_flag == 'hybrid':
            self.fc_1 = nn.Linear(input_dim * A, embed_dim, bias=False)
        else:
            self.fc_1 = nn.Linear(input_dim, embed_dim, bias=False)
        self.act = nn.ReLU()

        # Additional linear transformation after ReLU
        if dual_layer:
            self.fc_2 = nn.Linear(embed_dim, embed_dim, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.bmm(adj, seq)

        # First Layer --- SGC
        out = self.fc_1(aggr_c)
        # Out Activation
        out = self.act(out)

        return out


# =====================================================================================================================
# =====================================================================================================================
# Neural-UCB module

class Est_module(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(Est_module, self).__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        # Initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        # self.init_param_paper()

    def init_param_paper(self):
        fc_1_weights_diag = np.random.normal(loc=0, scale=(4 / self.hidden_size),
                                             size=(self.hidden_size // 2, self.input_dim // 2))
        fc_1_weights = np.zeros((self.hidden_size, self.input_dim))
        fc_1_weights[0:self.hidden_size // 2, 0:self.input_dim // 2] = fc_1_weights_diag
        fc_1_weights[self.hidden_size // 2:, self.input_dim // 2:] = fc_1_weights_diag
        self.fc1.weight = nn.Parameter(torch.from_numpy(fc_1_weights).float())

        fc_2_weights_half = np.random.normal(loc=0, scale=(2 / self.hidden_size), size=(1, self.hidden_size // 2))
        fc_2_weights = np.concatenate((fc_2_weights_half, -1 * fc_2_weights_half), axis=1)
        self.fc2.weight = nn.Parameter(torch.from_numpy(fc_2_weights).float())

    def forward(self, x):
        # First layer
        out = self.act(self.fc1(x))
        # Second layer
        out = self.fc2(out)

        return out


# =====================================================================================================================
# =====================================================================================================================


class GNN_Deep_Net(nn.Module):
    def __init__(self, A, input_dim, embed_dim, fc_hidden_size=500, JK_pattern='JK', PPR_m=None, dual_layer=False,
                 model_mode_flag=None):
        super(GNN_Deep_Net, self).__init__()
        self.fc_hidden_size = fc_hidden_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.JK_pattern = JK_pattern
        self.model_mode_flag = model_mode_flag
        self.PPR_m = PPR_m

        # Aggregation module
        if PPR_m is None:
            self.aggr = Aggr_module(A, input_dim, embed_dim, dual_layer=dual_layer, model_mode_flag=model_mode_flag)
        else:
            self.aggr = Aggr_module_PR(A, input_dim, embed_dim, PPR_m=PPR_m,
                                       dual_layer=dual_layer, model_mode_flag=model_mode_flag)

        # Estimation module
        if model_mode_flag == 'independent':
            self.est_module = Est_module(embed_dim=embed_dim + input_dim, hidden_size=fc_hidden_size)
        elif model_mode_flag == 'clustering':
            self.est_module = Est_module(embed_dim=embed_dim + input_dim, hidden_size=fc_hidden_size)
        elif model_mode_flag == 'hybrid':
            self.est_module = Est_module(embed_dim=embed_dim + (A * input_dim), hidden_size=fc_hidden_size)
        elif model_mode_flag == 'pooling':
            self.est_module = Est_module(embed_dim=embed_dim + input_dim, hidden_size=fc_hidden_size)

    def forward(self, extended_seq, origin_seq, adj, arm_seq):
        # overall_seq -> (batch_num, node_num, embed_dim)
        # h_1 -> same shape
        if self.model_mode_flag == 'hybrid':
            h_1 = self.aggr(origin_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'independent':
            h_1 = self.aggr(extended_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'clustering':
            h_1 = self.aggr(extended_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'pooling':
            h_1 = self.aggr(origin_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, origin_seq], dim=2)

        # Embedded contexts for the labeled arm
        # (batch_num=1, embed_dim)
        embed_c = h_c.squeeze(dim=0)
        embed_c = embed_c.index_select(0, arm_seq)

        # Point estimations
        point_ests = self.est_module(embed_c)

        # (1, 1)
        return point_ests

    #
    def predict(self, extended_seq, origin_seq, adj):
        # (1, node_num, embed_dim)
        if self.model_mode_flag == 'hybrid':
            h_1 = self.aggr(origin_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'independent':
            h_1 = self.aggr(extended_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'clustering':
            h_1 = self.aggr(extended_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, extended_seq], dim=2)
        elif self.model_mode_flag == 'pooling':
            h_1 = self.aggr(origin_seq, adj)
            # Concat after aggregation
            h_1 = F.normalize(h_1, dim=2)
            h_c = torch.cat([h_1, origin_seq], dim=2)

        # Normalize the embedded context
        h_c = h_c.squeeze(dim=0)

        # see_vec = h_c.detach().cpu().numpy()
        point_ests = self.est_module(h_c)

        return point_ests

