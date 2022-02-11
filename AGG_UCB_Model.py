import argparse

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from AGG_UCB_Network import GNN_Deep_Net
import sklearn.metrics.pairwise as Kernel


class AGG_UCB:
    # random_init: use random initialization or not
    def __init__(self, A, d, init_d, T, data_flag, gamma, hidden_size, bandwidth_c, bandwidth_m, threshold,
                 adj_diag_val=-1, PPR_m=None,
                 true_arm_cluster=None, whole_gradients=False, float_weights=False, JK_pattern=None, dual_layer=False,
                 MODEL_MODE=None):
        """
        :param d:       number of features of each item
        :param T:       Run the algorithm for T times
        :param edge_probability:
        """

        self.d = d
        self.init_d = init_d
        self.float_weights = float_weights
        self.model_mode_flag = MODEL_MODE

        self.bandwidth = bandwidth_c
        self.bandwidth_m = bandwidth_m if bandwidth_m > 0 else bandwidth_c
        self.threshold = threshold
        self.JK_pattern = JK_pattern
        self.PPR_m = PPR_m

        self.A = A
        self.arm_context_list_original = {i: [] for i in range(A)}
        self.arm_context_count = [0 for _ in range(A)]
        self.arm_reward_list = {i: [] for i in range(A)}

        # Data set type --- classification / recommendation
        if data_flag in (5, 7, 77):
            self.dataset_type = 'cla'
        else:
            self.dataset_type = 'rec'
        print("Data set type: ", self.dataset_type)

        # Number of clusters at time tx
        self.num_clusters = np.zeros(T)

        # ---- Kernel evaluations
        self.self_kernel_dis = {}
        self.mutual_kernel_dis = {}
        self.self_kernel_dis_sum = {}
        self.mutual_kernel_dis_sum = {}

        # ===================================================
        # Observation array
        self.time_reward_list = []

        self.all_arm_context_extended = []
        self.all_arm_context_original = []
        self.selected_arms_t = []

        # ----
        self.model = GNN_Deep_Net(A, input_dim=self.d, embed_dim=hidden_size, fc_hidden_size=hidden_size,
                                  JK_pattern=JK_pattern, PPR_m=PPR_m, dual_layer=dual_layer, model_mode_flag=MODEL_MODE)

        # Initialize the 'Z' matrix for CB
        self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = gamma * torch.ones((self.total_param,)).cuda()

        print("CB Gradient size: ", self.total_param)

        # True clustering
        if true_arm_cluster is not None:
            # True given arm graph
            self.ture_clustering_flag = True
            self.running_est_flag = False
            self.true_cluster = true_arm_cluster
            self.arm_G = true_arm_cluster
        else:
            # Clustering estimation
            self.ture_clustering_flag = False
            self.running_est_flag = True
            self.arm_G = nx.empty_graph(A, create_using=nx.Graph())

    def initialize_context(self, init_contexts, initial_short_contexts, init_rewards):
        # N == 1
        # User vectors
        for i in range(self.A):
            # (1, A, d)
            # all_contexts_ext = init_contexts[i, :, :].toarray()
            all_contexts_ori = initial_short_contexts[i, :, :]

            reward = init_rewards[i, i]

            self.arm_context_list_original[i].append(initial_short_contexts[i, i, :].reshape(-1, ))

            #
            self.arm_context_count[i] += 1
            self.arm_reward_list[i].append(reward)

            # Update all the arm context
            # Update all the arm context
            self.selected_arms_t.append([i])
            self.time_reward_list.append(reward)

            # append [A, d]
            self.all_arm_context_extended.append(init_contexts[i])
            self.all_arm_context_original.append(all_contexts_ori)

        if not self.ture_clustering_flag:
            # Init arm graph estimation
            self.initialize_arm_graph()

    # ==========================================================================
    def add_new_selected_info(self, t, observation, reward, arm_selected, all_c_contexts, all_ori_contexts,
                              stop_limit=np.inf, item_2_arm_dict=None, selected_item_2_arm=None):
        # Update arm graph ===========
        # AFTER CALCULATING THE LOSS
        if not self.ture_clustering_flag and stop_limit > t:

            if item_2_arm_dict is not None:
                item_2_arm_set = set()
                for i in range(len(item_2_arm_dict)):
                    item_2_arm_set = item_2_arm_set.union(set(item_2_arm_dict[i]))
                self.update_arm_graph(all_ori_contexts=all_ori_contexts, arm_selected=arm_selected,
                                      update_arm_set=item_2_arm_set, item_2_arm_dict=item_2_arm_dict)
        else:
            # End arm clustering estimation
            self.running_est_flag = False

        # Add original contexts to lists for similarity estimation
        # AFTER UPDATING GRAPH
        if self.running_est_flag:
            if item_2_arm_dict is None:
                for i in range(self.A):
                    self.arm_context_list_original[i].append(all_ori_contexts[i, :].reshape(-1, ))
                    self.arm_context_count[i] += 1
            else:
                for i in range(len(item_2_arm_dict)):
                    for arm_i in item_2_arm_dict[i]:
                        self.arm_context_count[arm_i] += 1
                        self.arm_context_list_original[arm_i].append(all_ori_contexts[i, :].reshape(-1, ))

        #
        self.arm_reward_list[arm_selected].append(reward)

        # Update all the arm context
        if selected_item_2_arm is None:
            self.selected_arms_t.append(arm_selected)
        else:
            self.selected_arms_t.append(selected_item_2_arm)
        self.time_reward_list.append(reward)

        # append [A, d]
        # sp_all_c_contexts = csr_matrix(all_c_contexts)
        # self.all_arm_context_extended.append(sp_all_c_contexts)
        self.all_arm_context_extended.append(all_c_contexts)
        self.all_arm_context_original.append(all_ori_contexts)

    # Arm graph estimation ops -------------------------------------------
    def initialize_arm_graph(self):
        # Initialize arm graph
        # One context per arm
        for a_i in range(self.A):
            self_m = Kernel.rbf_kernel(np.array(self.arm_context_list_original[a_i]),
                                       np.array(self.arm_context_list_original[a_i]),
                                       self.bandwidth)
            self.self_kernel_dis[a_i] = np.mean(self_m)
            self.self_kernel_dis_sum[a_i] = np.sum(self_m)

        for i in range(self.A):
            for j in range(i, self.A):
                mutual_m = Kernel.rbf_kernel(np.array(self.arm_context_list_original[i]),
                                             np.array(self.arm_context_list_original[j]), self.bandwidth)
                mean_val = np.mean(mutual_m)
                self.mutual_kernel_dis[tuple((i, j))] = mean_val
                self.mutual_kernel_dis_sum[tuple((i, j))] = np.sum(mutual_m)
                distance = self.self_kernel_dis[i] + self.self_kernel_dis[j] - 2 * mean_val

                # Arm similarity measurement
                simi_val = np.exp(-self.bandwidth_m * distance)
                if simi_val > self.threshold:
                    weight = simi_val if self.float_weights else 1
                    self.arm_G.add_edge(i, j, weight=weight)

        print("Initialized estimated arm graph")

    def update_arm_graph(self, all_ori_contexts, arm_selected=None, update_arm_set=None, item_2_arm_dict=None):

        # ------------------------------------------------------------------------------------------------------------
        # kernel mean embedding --- Global graph update --- recommendation data set
        update_arm_list = update_arm_set
        # New contexts
        new_arm_context = {i: [] for i in range(self.A)}
        for i, arm_list in item_2_arm_dict.items():
            for arm in arm_list:
                new_arm_context[arm].append(all_ori_contexts[i, :].reshape(-1, ))

        # Arm self update
        for a_i in range(self.A):
            if a_i not in update_arm_list:
                continue
            # construct new matrix
            new_c = np.array(new_arm_context[a_i])
            new_row = Kernel.rbf_kernel(new_c, np.array(self.arm_context_list_original[a_i]), self.bandwidth)

            new_diag = Kernel.rbf_kernel(new_c, new_c, self.bandwidth)
            new_total = self.self_kernel_dis_sum[a_i] + (2 * np.sum(new_row)) + np.sum(new_diag)
            self.self_kernel_dis_sum[a_i] = new_total
            self.self_kernel_dis[a_i] = new_total / ((self.arm_context_count[a_i] + len(new_arm_context[a_i])) ** 2)

        # Pair-wise similarity
        for i in range(self.A):
            for j in range(i + 1, self.A):
                if i not in update_arm_list and j not in update_arm_list:
                    continue
                #
                if i in update_arm_list:
                    new_c_i = np.array(new_arm_context[i])
                    new_row = np.sum(Kernel.rbf_kernel(new_c_i, np.array(self.arm_context_list_original[j]),
                                                       self.bandwidth))
                else:
                    new_c_i = None
                    new_row = 0
                #
                if j in update_arm_list:
                    new_c_j = np.array(new_arm_context[j])
                    new_col = np.sum(Kernel.rbf_kernel(new_c_j, np.array(self.arm_context_list_original[i]),
                                                       self.bandwidth))
                else:
                    new_c_j = None
                    new_col = 0
                #
                if i in update_arm_list and j in update_arm_list:
                    new_diag = np.sum(Kernel.rbf_kernel(new_c_i, new_c_j, self.bandwidth))
                else:
                    new_diag = 0

                #
                new_total = self.mutual_kernel_dis_sum[tuple((i, j))] + (new_row + new_col) + new_diag

                #
                mutual_val = new_total / \
                             ((self.arm_context_count[i] + len(new_arm_context[i])) *
                              (self.arm_context_count[j] + len(new_arm_context[j])))

                self.mutual_kernel_dis[tuple((i, j))] = mutual_val
                self.mutual_kernel_dis_sum[tuple((i, j))] = new_total

                # Arm similarity measurement
                distance = self.self_kernel_dis[i] + self.self_kernel_dis[j] - 2 * mutual_val
                simi_val = np.exp(-self.bandwidth_m * distance)

                # Update edge weights
                if self.threshold > 0:
                    if simi_val > self.threshold:
                        # Add edge
                        weight = simi_val if self.float_weights else 1
                        self.arm_G.add_edge(i, j, weight=weight)
                    else:
                        # Delete edge
                        if self.arm_G.has_edge(i, j):
                            self.arm_G.remove_edge(i, j)
                else:
                    # Update the edge weights only
                    self.arm_G[i][j]['weight'] = simi_val if self.float_weights else 1
        print("Updated arm graph Rec...")

    # ================= Embed the contexts into graphs
    def get_embed_arm_graph_batches(self, window_size=-1, SAMPLE_TRAIN_NUM=1000):
        time_length = len(self.all_arm_context_original)

        # Select the latest time steps
        if 0 < window_size < time_length:
            choices = np.arange(time_length)[-window_size:]
            batch_matrix_extended = self.all_arm_context_extended[-window_size:]
            # batch_matrix_extended = np.array(self.all_arm_context_extended[-window_size:], dtype=np.float)
            batch_matrix_original = np.array(self.all_arm_context_original[-window_size:], dtype=np.float)
            reward_array = np.array(self.time_reward_list, dtype=np.float).reshape((-1, 1))[choices, :]
            time_length = batch_matrix_original.shape[0]
            arm_array = [self.selected_arms_t[i] for i in choices]
            I_permutation = np.random.permutation(time_length)
        # Whole data set
        else:
            if SAMPLE_TRAIN_NUM > 0:
                choices = np.random.choice(time_length, size=(min(SAMPLE_TRAIN_NUM, time_length),), replace=False) \
                    .tolist()
                ori_c_list = [self.all_arm_context_original[i] for i in choices]
                batch_matrix_extended = np.array([self.all_arm_context_extended[i] for i in choices], dtype=np.float)
                batch_matrix_original = np.array(ori_c_list, dtype=np.float)
                reward_array = np.array(self.time_reward_list, dtype=np.float).reshape((-1, 1))[choices, :]
                time_length = batch_matrix_original.shape[0]
                arm_array = [self.selected_arms_t[i] for i in choices]
                I_permutation = np.random.permutation(time_length)
            else:
                # (time_steps, node_num, dim)
                batch_matrix_extended = self.all_arm_context_extended
                # batch_matrix_extended = np.array(self.all_arm_context_extended, dtype=np.float)
                batch_matrix_original = np.array(self.all_arm_context_original, dtype=np.float)
                reward_array = np.array(self.time_reward_list, dtype=np.float).reshape((-1, 1))
                arm_array = self.selected_arms_t
                I_permutation = np.random.permutation(time_length)

        return batch_matrix_extended, batch_matrix_original, time_length, reward_array, arm_array, I_permutation
