from __future__ import division
import numpy as np
import sklearn.metrics.pairwise as Kern
import DataCreate.DataCreate as DC
import KernelCalculation.GaussianKernels as GK
import scipy
from scipy import special as special
import math
from sklearn.preprocessing import normalize

K_SIM_FLAG = False
S_A_T_FLAG = False
CHANGE_TO_NEW_SIMILARITY = 0
SIMILARITY_PARAMETER = 0.99
IF_SOFTMAX = False

# Flag for enabling additional estimation from Clustering
ENABLE_ADDITIONAL_ESTIMATE = True

# This function runs the bandit algorithm at time t
def ContextBanditUCB(DataXY, tt, bw_x, bw_prod, gamma, bw_prob, alpha, data_flag, algorithm_flag):
    """
    :param DataXY:
    :param tt:  time stamp t
    :param bw_x:
    :param bw_prod:
    :param gamma:
    :param bw_prob:
    :param alpha: -> Lemma 1
    :param data_flag:
    :param algorithm_flag:
    :return:
    """

    # Dict for post matrix elements
    affinity_dict = DataXY['affinity_dict']
    simi_dict = DataXY['simi_dict']

    # KTLUCB
    total_samples, samples_per_task, y, X_total = DC.AllDataCollect(DataXY, algorithm_flag)

    # total_samples -> samples_per_task * num_arms,
    # samples_per_task -> samples_per_arm,
    # y -> labels,
    # X_total -> total_dataset ---- (N*A, 2)

    samples_per_task = samples_per_task.astype(int)
    A = DataXY['NoOfArms']
    # theta[aa] = aa * (pi / 25), theta[0] = 0.0
    theta = DataXY['theta']

    # theta -> multi-class = 0
    K_sim, eta_arm, Task_sim, DataXY= GK.GetKernelMatrix(DataXY, X_total, A, total_samples, samples_per_task,
                                                                    bw_x,
                                                                    theta, bw_prod, algorithm_flag, bw_prob, gamma, tt)


    # K_sim ========== capital K_tilde --- total_samples x total_samples
    #                                  --- N*A x N*A
    # eta_arm -> diagonal matrix

    # -----------------------------------------------------------------

    # Run the UCB estimate
    # KRR Estimate using training data and direct inverse

    if algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
        eta = np.diag(eta_arm)
        # Inverse the matrix
        InvTerm = np.linalg.inv(eta.dot(K_sim) + gamma * np.identity(K_sim.shape[0]))

    # Reward for the clustering ---------
    reward_cluster = np.zeros([A, 1])

    # ------------------------------------------- Run the algorithm at time t

    feature_list = np.zeros([A, DataXY['d']])

    # Arm iterations ==============================================
    if data_flag == 5 or data_flag == 7 or data_flag == 77:
        reward = np.zeros([A, 1])
        reward_est = np.zeros([A])
        for aa in range(0, A):
            # For real-world dataset
            Features_test = DataXY['TestContexts']
            X_test = Features_test[tt, :].reshape(1, -1)
            X_test = normalize(X_test)

            if algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
                # K_x ===> k_(t-1) (x)  ------ shape (N*A + t, 1)
                K_x = np.zeros([X_total.shape[0], X_test.shape[0]])
                rr = 0
                for i in range(0, A):
                    Xi = X_total[rr:rr + samples_per_task[i], :]
                    # k_a,t ---- Evaluating the user context with past data

                    K_x[rr:rr + samples_per_task[i], :] = Task_sim[i, aa] * Kern.rbf_kernel(Xi, X_test, bw_x)
                    rr = rr + samples_per_task[i]

                # k_tilde (x_(a,t), x_(a,t))-------------------
                k_x_a = Kern.rbf_kernel(X_test, X_test, bw_x)

                # f^hat (x_tilde)---------------------------------------------- y => N*A + t (labels for samples)
                reward_est[aa] = np.transpose(K_x).dot(InvTerm).dot(eta).dot(y)

                # square of s_(a,t) ----------------------------------------------
                reward_conf = k_x_a - np.transpose(K_x).dot(InvTerm).dot(eta).dot(K_x)
                sqrt_reward_conf = np.sqrt(reward_conf)

            # ===================================================================

            if tt % 20 == 0 and algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
                print("Est: ", reward_est[aa])
                print("Conf: ", alpha * sqrt_reward_conf)

            reward[aa] = reward_est[aa] + alpha * sqrt_reward_conf

    # Item iterations ==============================================
    elif data_flag == 10 or data_flag == 12:
        item_pool_size = DataXY['item_pool_size']
        category_dict = DataXY['Category_Dict']
        pool_sample_size = 0
        context_2_arm_list = []
        context_2_item_list = []
        for i in range(item_pool_size):
            arm_list_i = category_dict[tuple([tt, i])]
            for arm in arm_list_i:
                context_2_arm_list.append(arm)
                context_2_item_list.append(i)
                pool_sample_size += 1

        reward = np.zeros([pool_sample_size, 1])
        reward_est = np.zeros([pool_sample_size])
        print("Pool size: ", pool_sample_size)

        for c_i in range(0, pool_sample_size):
            # For real-world dataset
            context_matrix = DataXY['context_matrix']
            item_i = context_2_item_list[c_i]
            aa = context_2_arm_list[c_i]

            X_test = context_matrix[tt, item_i, :].reshape(1, -1)
            X_test = normalize(X_test)

            if algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
                # K_x ===> k_(t-1) (x)  ------ shape (N*A + t, 1)
                K_x = np.zeros([X_total.shape[0], X_test.shape[0]])
                rr = 0
                for i in range(0, A):
                    Xi = X_total[rr:rr + samples_per_task[i], :]
                    # k_a,t ---- Evaluating the user context with past data

                    K_x[rr:rr + samples_per_task[i], :] = Task_sim[i, aa] * Kern.rbf_kernel(Xi, X_test, bw_x)
                    rr = rr + samples_per_task[i]

                # k_tilde (x_(a,t), x_(a,t))-------------------
                k_x_a = Kern.rbf_kernel(X_test, X_test, bw_x)

                # f^hat (x_tilde)---------------------------------------------- y => N*A + t (labels for samples)
                reward_est[c_i] = np.transpose(K_x).dot(InvTerm).dot(eta).dot(y)

                # square of s_(a,t) ----------------------------------------------
                reward_conf = k_x_a - np.transpose(K_x).dot(InvTerm).dot(eta).dot(K_x)
                sqrt_reward_conf = np.sqrt(reward_conf)

            # ===================================================================

            if tt % 20 == 0 and algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
                print("Est: ", reward_est[c_i])
                print("Conf: ", alpha * sqrt_reward_conf)

            reward[c_i] = reward_est[c_i] + alpha * sqrt_reward_conf
    #
    else:
        exit(1)

    # Choose an arm with maximum p_a,t ------------------------------------
    if data_flag == 5 or data_flag == 7 or data_flag == 77:
        selected_arm = np.argmax(reward)
        Features_test = DataXY['TestContexts']
        X_test = Features_test[tt, :].reshape(1, -1)
        X_test = normalize(X_test)

    elif data_flag == 10:
        selected_arm = np.argmax(reward)
        item_i = context_2_item_list[selected_arm]

        X_test = context_matrix[tt, item_i, :].reshape(1, -1)
        X_test = normalize(X_test)

    elif data_flag == 11:
        userContext = DataXY['userContext']
        armContext = DataXY['armContext']

        selected_arm = np.argmax(reward)

        userVec = userContext[tt, :].reshape(-1, 1)
        armVec = armContext[selected_arm, :].reshape(-1, 1)

        X_test = np.multiply(userVec, armVec).T
    elif data_flag == 12:
        selected_arm = np.argmax(reward)
        item_i = context_2_item_list[selected_arm]

        X_test = context_matrix[tt, item_i, :].reshape(1, -1)
        X_test = normalize(X_test)

    elif data_flag == 13:
        userContext = DataXY['userContext']
        armContext = DataXY['armContext']

        selected_arm = np.argmax(reward)

        userVec = userContext[tt, :].reshape(-1, 1)
        armVec = armContext[selected_arm, :].reshape(-1, 1)

        X_test = np.multiply(userVec, armVec).T

    elif data_flag == 99:
        userContext = DataXY['userContext']
        armContext = DataXY['armContext']

        selected_arm = np.argmax(reward)

        userVec = userContext[tt, :].reshape(-1, 1)
        armVec = armContext[selected_arm, :].reshape(-1, 1)

        X_test = np.multiply(userVec, armVec).T

    elif data_flag == 100:
        selected_arm = np.argmax(reward)
        X_test = feature_list[selected_arm, :].reshape(1, -1)

    elif data_flag == 101:
        userContext = DataXY['userContext']
        armContext = DataXY['armContext']

        selected_arm = np.argmax(reward)

        userVec = userContext[tt, :].reshape(-1, 1)
        armVec = armContext[selected_arm, :].reshape(-1, 1)

        X_test = np.multiply(userVec, armVec).T

    return reward, reward_cluster, X_test, DataXY
