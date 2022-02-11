import argparse

import networkx as nx
import numpy as np

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from AGG_UCB_Network import GNN_Deep_Net
from AGG_UCB_Model import AGG_UCB
import AGG_UCB_Utils as utils

import time

from datetime import datetime
import sys


# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open("./AGG-UCB-logs/logfile_" + dt_string + "_AGG-UCB_.log", "w")
        self.out = stdout
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        pass


sys.stdout = Logger(sys.stdout)

if __name__ == '__main__':
    # =======================
    torch.cuda.set_device(0)
    # =======================

    # ============================
    # Since the XRMB data set is quite large (1 GiB ~), if you want to apply the data set,
    # please download the files from (https://home.ttic.edu/~klivescu/XRMB_data/full/README)
    # and put these two data files in the './Dataset' folder.
    # ============================

    # ======================================================================== MovieLens data set
    # # Data set
    data_flag = 12

    # Learning rate
    init_lr = 1e-3

    # Sample the past contexts for training, INSTEAD OF FULL TRAINING
    SAMPLE_TRAIN_NUM = 1000
    train_epoch_num = 20

    #
    Est_GRAPH_STOP_LIMIT = 4000

    #
    nu = 0.0001

    # --- Bandwidth for similarity
    bandwidth = 10
    # BW_kernel mean embedding --- NO USE WHEN TRUE CLUSTERING
    bandwidth_m = 2e3

    # k-th neighborhood
    k = 1

    # ======================================================================== Yelp data set
    # # # Data set
    # data_flag = 10
    #
    # # Learning rate
    # init_lr = 1e-3
    #
    # # Sample the past contexts for training, INSTEAD OF FULL TRAINING
    # SAMPLE_TRAIN_NUM = 1000
    # train_epoch_num = 20
    #
    # #
    # Est_GRAPH_STOP_LIMIT = 4000
    #
    # #
    # nu = 0.001
    #
    # # --- Bandwidth for similarity
    # bandwidth = 50
    # # BW_kernel mean embedding --- NO USE WHEN TRUE CLUSTERING
    # bandwidth_m = 1e3
    #
    # # k-th neighborhood
    # k = 1

    # # ======================================================================== XRMB data set
    # # # Data set
    # data_flag = 5
    #
    # # Learning rate
    # init_lr = 1e-3
    #
    # # Sample the past contexts for training, INSTEAD OF FULL TRAINING
    # SAMPLE_TRAIN_NUM = 300
    # train_epoch_num = 20
    #
    # #
    # Est_GRAPH_STOP_LIMIT = 2000
    #
    # #
    # nu = 0.01
    #
    # # --- Bandwidth for similarity
    # bandwidth = 5
    # # BW_kernel mean embedding --- NO USE WHEN TRUE CLUSTERING
    # bandwidth_m = 100
    #
    # # k-th neighborhood
    # k = 1

    # ======================================================================== MNIST Augmented data set
    # Data set
    # data_flag = 77
    #
    # # Learning rate
    # init_lr = 1e-3
    #
    # # Sample the past contexts for training, INSTEAD OF FULL TRAINING
    # SAMPLE_TRAIN_NUM = 300
    # train_epoch_num = 20
    #
    # #
    # Est_GRAPH_STOP_LIMIT = 2000
    #
    # #
    # nu = 0.001
    #
    # # --- Bandwidth for similarity
    # bandwidth = 5
    # # BW_kernel mean embedding --- NO USE WHEN TRUE CLUSTERING
    # bandwidth_m = 100
    #
    # # k-th neighborhood
    # k = 1
    # ==================================================================================================================
    # ==================================================================================================================

    # Shifting window
    WINDOW_SIZE = -1

    # "hybrid", "independent", "pooling", "clustering"
    Model_mode = 'independent'
    cluster_num = 10

    # Network
    FC_hidden_size = 500
    # Confidence coefficient

    # Feed with true arm clustering
    TRUE_CLUSTERING = False

    # Edge threshold
    threshold = -1

    # Float-type edge weights
    FLOAT_WEIGHT = True

    # Basic param --- fixed
    N, N_valid = 1, 10
    T = 4000
    #
    lambda_val = 0.001

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    if data_flag == 5:
        data_flag_multiclass = 'real-world_XRMB'
        A = 38
    elif data_flag == 77:
        data_flag_multiclass = 'real-world_MNIST_Augmented_Labels'
        A = 50
    elif data_flag == 10:
        data_flag_multiclass = 'real-world_Yelp'
        A = 20
    elif data_flag == 12:
        data_flag_multiclass = 'real-world_MovieLens'
        A = 19
    else:
        data_flag_multiclass = None
        A = None
    print("Data set: ", data_flag_multiclass)
    WHOLE_GRADIENTS = True
    # ==================================================================================================================
    #
    Basic_DataXY, input_dim, init_dim, arm_G, true_graph, init_X, init_X_short, init_reward_m, cluster_num, \
    X, X_short, rewards_m, \
        arm_2_cluster_dict \
        = utils.get_datasets_and_rewards(A, T, N, N_valid, threshold, bandwidth, cluster_num=cluster_num,
                                         data_flag=data_flag, TRUE_CLUSTERING=False, Model_mode=Model_mode,
                                         FLOAT_WEIGHT=FLOAT_WEIGHT, NEW_SELF_LOOP_VAL=1)

    # ==================================================================================================================
    # --- Model
    PPR_m = None

    #
    mdu = AGG_UCB(A, input_dim, init_dim, T, data_flag, lambda_val, hidden_size=FC_hidden_size,
                       bandwidth_c=bandwidth, bandwidth_m=bandwidth_m,
                       threshold=threshold, adj_diag_val=-1, PPR_m=PPR_m,
                       true_arm_cluster=arm_G, whole_gradients=WHOLE_GRADIENTS, float_weights=FLOAT_WEIGHT,
                       MODEL_MODE=Model_mode)
    # N = 1
    mdu.initialize_context(init_contexts=init_X, initial_short_contexts=init_X_short,
                           init_rewards=init_reward_m)
    mdu.model.cuda()

    # Train param
    BATCH_SIZE = 1

    lr_decay_step = 1000
    lr_decay_rate = 0.7

    print("Param: ", bandwidth, threshold, bandwidth_m, TRUE_CLUSTERING, Model_mode, WINDOW_SIZE,
          FC_hidden_size, nu, FLOAT_WEIGHT, Est_GRAPH_STOP_LIMIT, init_lr, k)

    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(mdu.model.parameters(), lr=init_lr, weight_decay=lambda_val)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)

    #
    true_adj_matrix = np.array(nx.adjacency_matrix(true_graph).todense())
    true_adj_matrix_normalized = utils.get_sym_norm_matrix(source=true_adj_matrix)
    if k > 1:
        true_adj_matrix_normalized = np.linalg.matrix_power(true_adj_matrix_normalized, k)
    with np.printoptions(threshold=np.inf):
        print('-' * 20)
        print("Initial adj matrix: \n")
        _adj = np.array(nx.adjacency_matrix(mdu.arm_G).todense())
        print(_adj)
        print("Current edges: ", len(mdu.arm_G.edges))
        if Model_mode == 'clustering':
            print("Cluster num: ", cluster_num)
        print('-' * 20)

    # =====================================================================================
    # ================================= Run the algorithm =================================
    acc_num = 0
    cumulative_regret = 0.0
    last_centers = None
    s_time = time.time()

    for t in range(T):

        step_time = time.time()

        #
        adj_matrix = nx.adjacency_matrix(mdu.arm_G).todense()
        adj_matrix = utils.get_sym_norm_matrix(source=adj_matrix)
        if k > 1:
            adj_matrix = np.linalg.matrix_power(adj_matrix, k)
        sym_adj_m = adj_matrix
        adj_single = torch.tensor(adj_matrix[np.newaxis, :, :]).float().cuda()

        #
        if data_flag == 10 or data_flag == 12:
            print("Get new long embedded contexts categories...")
            context_list = []
            rwd_list = []
            items_per_step = X_short.shape[1]
            for a_i in range(items_per_step):
                normalized_vec = X_short[t, a_i, :]
                category_list = Basic_DataXY['Category_Dict'][tuple([t, a_i])]
                embed_contexts = utils.generate_long_vec_category(normalized_vec, A)
                context_list.append(embed_contexts)
                for _ in range(len(category_list)):
                    rwd_list.append(float(rewards_m[t, a_i]))
            rwd = np.array(rwd_list).reshape(-1, )
            print("This pool size: ", len(rwd_list))
        #
        c_contexts_short = X_short[t, :, :]

        mdu.model.eval()
        # ================================== Make predictions for current contexts ==================================
        # Neural point estimations

        estimations = []
        g_list = []
        # Reward estimations
        ss_time = time.time()
        if data_flag == 10 or data_flag == 12:
            items_per_step = X_short.shape[1]
            context_2_arm_list = []
            context_2_item_list = []
            #
            for item_i in range(items_per_step):
                # (1, node_num, d)
                this_context = context_list[item_i][np.newaxis, :, :]
                ext_inputs = torch.tensor(this_context).float().cuda()
                ori_inputs = torch.tensor(c_contexts_short[np.newaxis, :, :]).float().cuda()

                # Point Ests
                mdu.model.zero_grad()
                point_ests = mdu.model.predict(ext_inputs, ori_inputs, adj_single)

                # Confidence bound
                for e_i, p_est in enumerate(point_ests):
                    # Skip false categories
                    if e_i not in Basic_DataXY['Category_Dict'][tuple([t, item_i])]:
                        continue
                    # Add in the category
                    context_2_arm_list.append(e_i)
                    context_2_item_list.append(item_i)

                    # Backprop for this point estimation
                    mdu.model.zero_grad()
                    p_est.backward(retain_graph=True)

                    # Try the gradient of the whole framework
                    g = torch.cat([p.grad.flatten().detach() for p in mdu.model.parameters() if p.requires_grad])
                    g_list.append(g)

                    # CB
                    cb_sq = lambda_val * nu * g * g / mdu.U
                    cb = torch.sqrt(torch.sum(cb_sq))

                    # Full estimation
                    full_est = p_est.item() + cb.item()
                    # print(full_est)
                    estimations.append(full_est)

            # print(estimations)
            selected_item = int(np.argmax(estimations))
            print("--- Selected item: ", selected_item)
            g_prod = g_list[selected_item] * g_list[selected_item]
            mdu.U += g_prod

            # True background reward
            t_reward = rwd[selected_item]
            best_item = np.argwhere(rwd == np.max(rwd)).flatten()
            if selected_item in best_item:
                acc_num += 1

            # Regret
            this_regret = np.max(rwd) - t_reward
            cumulative_regret += this_regret

            # For updating the context
            c_contexts = context_list[context_2_item_list[selected_item]]
            selected_arm = context_2_arm_list[selected_item]
            selected_item_2_arm = Basic_DataXY['Category_Dict'][tuple([t, context_2_item_list[selected_item]])]
            best_arm = best_item
            item_2_arm_dict = {i: Basic_DataXY['Category_Dict'][tuple([t, i])] for i in range(items_per_step)}
        #

        elif data_flag in (5, 7, 77):
            # (1, node_num, d)
            normalized_vec = c_contexts_short[0, :]
            c_contexts = c_contexts_short
            embedded_context = utils.generate_long_vec_category(normalized_vec, A)

            ext_inputs = torch.tensor(embedded_context[np.newaxis, :, :]).float().cuda()
            ori_inputs = torch.tensor(c_contexts_short[np.newaxis, :, :]).float().cuda()
            point_ests = mdu.model.predict(ext_inputs, ori_inputs, adj_single)
            # Confidence bound
            for i, p_est in enumerate(point_ests):
                # Backprop for this point estimation
                mdu.model.zero_grad()
                p_est.backward(retain_graph=True)

                # Try the gradient of the whole framework
                if WHOLE_GRADIENTS:
                    g = torch.cat([p.grad.flatten().detach() for p in mdu.model.parameters() if p.requires_grad])
                    g_list.append(g)
                else:
                    # Get gradients for the two FC layers
                    g = torch.cat([p.grad.flatten().detach() for p in mdu.model.est_module.parameters()
                                   if p.requires_grad])
                    g_list.append(g)

                # CB
                cb_sq = lambda_val * nu * g * g / mdu.U
                cb = torch.sqrt(torch.sum(cb_sq))

                # Full estimation
                full_est = p_est.item() + cb.item()
                # print(full_est)
                estimations.append(full_est)

            # print(estimations)
            selected_arm = int(np.argmax(estimations))
            g_prod = g_list[selected_arm] * g_list[selected_arm]
            mdu.U += g_prod

            # True background reward
            t_reward = rewards_m[t, selected_arm]
            rwd_list = rewards_m[t, :]
            best_arm = np.argwhere(rwd_list == np.max(rwd_list)).flatten()
            if selected_arm in best_arm:
                acc_num += 1

            # Regret
            this_regret = np.max(rwd_list) - t_reward
            cumulative_regret += this_regret

            # One arm to one item
            context_2_arm_list = None
            selected_item_2_arm = [selected_arm]
            item_2_arm_dict = {0: [selected_arm]}

        print("predicting time: {}".format(time.time() - ss_time))

        # ================================== Save info
        b_s_time = time.time()
        mdu.add_new_selected_info(t=t, observation=c_contexts[selected_arm, :], reward=t_reward,
                                  arm_selected=selected_arm, all_c_contexts=c_contexts,
                                  all_ori_contexts=c_contexts_short, stop_limit=Est_GRAPH_STOP_LIMIT,
                                  item_2_arm_dict=item_2_arm_dict, selected_item_2_arm=selected_item_2_arm)
        print("Add info time: {}".format(time.time() - b_s_time))

        # ================================== Train the model ==================================
        mdu.model.train()

        # Get past data and shuffle
        b_s_time = time.time()
        batch_matrix_extended, batch_matrix_original, time_length, reward_array, arm_selected, I_permutation \
            = mdu.get_embed_arm_graph_batches(window_size=WINDOW_SIZE, SAMPLE_TRAIN_NUM=SAMPLE_TRAIN_NUM)

        # ==========
        if data_flag in (5, 7, 77):
            length, sample_dim = batch_matrix_extended.shape[0], batch_matrix_extended.shape[2]
            embedded_batch_matrix = np.zeros([length, A, sample_dim * A])
            for i in range(length):
                embedded_batch_matrix[i, :, :] = utils.generate_long_vec_category(batch_matrix_extended[i, 0, :], A)
        # ==========
        print("Get batch time: {}".format(time.time() - b_s_time))
        print("Time length: ", WINDOW_SIZE, time_length)

        step_loss = 0.0

        # Update adj matrix after deriving new arm graphs ------
        adj_matrix = nx.adjacency_matrix(mdu.arm_G).todense()
        adj_matrix = utils.get_sym_norm_matrix(source=adj_matrix)
        if k > 1:
            adj_matrix = np.linalg.matrix_power(adj_matrix, k)
        # if PPR_alpha > 0:
        #     adj_matrix = PPR_m

        # ==================================================================
        # ----------------------- Full size training -----------------------
        ss_time = time.time()
        if SAMPLE_TRAIN_NUM <= 0:
            for epoch in range(train_epoch_num):
                scheduler.step(epoch=epoch)
                epoch_loss = 0.0

                for index in range(0, time_length, BATCH_SIZE):
                    if data_flag in (5, 7, 77):
                        inputs_ext = torch.tensor(
                            np.asarray([embedded_batch_matrix[i, :, :]
                                        for i in I_permutation[index:index + BATCH_SIZE]])
                        ).float().cuda()
                    else:
                        inputs_ext = torch.tensor(
                            np.asarray([batch_matrix_extended[i, :, :]
                                        for i in I_permutation[index:index + BATCH_SIZE]])
                        ).float().cuda()

                    inputs_ori = torch.tensor(
                        np.asarray([batch_matrix_original[i, :, :] for i in I_permutation[index:index + BATCH_SIZE]])
                    ).float().cuda()

                    this_batch_size = min(inputs_ext.shape[0], BATCH_SIZE)

                    adj_input = torch.tensor(adj_matrix).repeat(this_batch_size, 1, 1).float().cuda()

                    arm_selected_batch = np.asarray([arm_selected[i] for i in I_permutation[index:index + BATCH_SIZE]])
                    # arm_indices = torch.tensor(arm_selected_batch, dtype=torch.long).cuda()

                    arm_indices = torch.tensor(arm_selected_batch[0], dtype=torch.long).cuda()
                    #
                    logits = mdu.model(inputs_ext, inputs_ori, adj_input, arm_indices)

                    t_rewards = torch.tensor(np.asarray(
                        [reward_array[i] for i in I_permutation[index:index + BATCH_SIZE]])) \
                        .float().cuda()

                    loss = mse_loss(logits, t_rewards)
                    epoch_loss += loss.item()
                    step_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # ----------------------- Sampling Training ---------------------------
        else:
            past_context_counter, ter_flag = 0, False
            for epoch in range(train_epoch_num):
                scheduler.step(epoch=epoch)
                epoch_loss = 0.0

                for index in range(0, time_length, BATCH_SIZE):

                    if data_flag in (5, 7, 77):
                        inputs_ext = torch.tensor(
                            np.asarray([embedded_batch_matrix[i, :, :]
                                        for i in I_permutation[index:index + BATCH_SIZE]])
                        ).float().cuda()
                    else:
                        inputs_ext = torch.tensor(
                            np.asarray([batch_matrix_extended[i, :, :]
                                        for i in I_permutation[index:index + BATCH_SIZE]])
                        ).float().cuda()

                    inputs_ori = torch.tensor(
                        np.asarray([batch_matrix_original[i, :, :] for i in I_permutation[index:index + BATCH_SIZE]])
                    ).float().cuda()

                    this_batch_size = min(inputs_ext.shape[0], BATCH_SIZE)

                    adj_input = torch.tensor(adj_matrix).repeat(this_batch_size, 1, 1).float().cuda()

                    arm_selected_batch = np.asarray([arm_selected[i] for i in I_permutation[index:index + BATCH_SIZE]])

                    # print(arm_selected_batch[0])
                    # arm_indices = torch.tensor(arm_selected_batch, dtype=torch.long).cuda()
                    arm_indices = torch.tensor(arm_selected_batch[0], dtype=torch.long).cuda()

                    logits = mdu.model(inputs_ext, inputs_ori, adj_input, arm_indices)

                    # t_rewards = torch.tensor(np.asarray(
                    #     [reward_array[i] for i in I_permutation[index:index + BATCH_SIZE]])) \
                    #     .float().cuda()
                    t_rewards = torch.tensor(np.asarray(
                        [reward_array[I_permutation[index]]] * logits.shape[0])) \
                        .float().cuda()

                    loss = mse_loss(logits, t_rewards)
                    # print("logits: ", logits)
                    # print("t_rewards: ", t_rewards)
                    # print("loss: ", loss)

                    epoch_loss += loss.item()
                    step_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #
                    past_context_counter += 1
                    if past_context_counter > SAMPLE_TRAIN_NUM:
                        ter_flag = True
                        break

                # Early stop
                if epoch_loss / time_length <= 1e-3 and TRUE_CLUSTERING:
                    print("!Early stop. Avg loss: ", epoch_loss / time_length)
                    break

                # Break flag
                if ter_flag:
                    break

        print("Training time: {}".format(time.time() - ss_time))

        # Latest adj matrix
        new_adj = nx.adjacency_matrix(mdu.arm_G).todense()
        new_adj = utils.get_sym_norm_matrix(source=new_adj)
        # if PPR_alpha > 0:
        #     new_adj = PPR_m

        print("-" * 20)
        print("Time: {} / {}, Acc: {}, c_regret: {}, \nBest arm: {}, Selected_Arm: {}, \nb_rwd: {}, f_hat: {}, "
              "CB: {}, Overall: {}"
              .format(t, T, acc_num / (t + 1), cumulative_regret, best_arm, selected_arm,
                      np.max(rwd_list),
                      point_ests[selected_arm, 0],
                      estimations[selected_arm] - point_ests[selected_arm, 0],
                      estimations[selected_arm]))
        if SAMPLE_TRAIN_NUM > 0:
            print("Training loss:", step_loss / SAMPLE_TRAIN_NUM)
        else:
            print("Training loss:", step_loss / (t + 1))

        print("Current edges: ", len(mdu.arm_G.edges))
        powered_mat = np.linalg.matrix_power(new_adj, max(1, k))
        powered_mat_norm = np.linalg.matrix_power(adj_matrix, max(1, k))
        print("Powered non-0 entries: ", np.count_nonzero(powered_mat), " / ", A ** 2)
        off_diag_matrix = np.copy(powered_mat)
        np.fill_diagonal(off_diag_matrix, 0)
        print("Off-diag Weight max:", np.max(off_diag_matrix))
        print("Off-diag Weight avg:", np.sum(off_diag_matrix) / (np.count_nonzero(off_diag_matrix)))
        print("Normed Weight max:", np.max(powered_mat_norm))
        print("Normed Weight avg:", np.sum(powered_mat_norm) / (np.count_nonzero(powered_mat_norm)))
        print("Time elapsed: {}, time for this step: {}".format(time.time() - s_time, time.time() - step_time))
        print(estimations)

        # -----------------------------------------------
