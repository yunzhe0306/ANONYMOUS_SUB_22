from __future__ import division
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics.pairwise as Kern
from scipy.special import softmax
import DataCreate.DataCreate as DC


def GetKernelMatrix(DataXY, X_total, A, total_samples, samples_per_task,
                    bw_x, theta, bw_prod,
                    algorithm_flag, bw_prob, gamma, tt):
    """
    :param DataXY:
    :param X_total:  shape (N + t, 2)
    :param A:
    :param total_samples:  ------   N x A
    :param samples_per_task:
    :param bw_x:
    :param theta:
    :param bw_prod:
    :param algorithm_flag:
    :param bw_prob:
    :param gamma:
    :return:
    """

    """
    (1) Parameters to update:
    def update_lists(self, arm_index, S, b):
    def update_arm_selected(self, arm_index):
    def update_user_context(self, arm_index, user_context):
    def update_clusters(self, t):
    
    (2) Get cluster Context:
    def get_cluster_context_for_arm
    """

    Task_sim = np.zeros([A, A])

    # Dict for post matrix elements
    affinity_dict = DataXY['affinity_dict']
    simi_dict = DataXY['simi_dict']
    if tt == 0:
        last_arm = []
    else:
        last_arm = DataXY['selected_arm'][tt - 1]

    # Generate embeddings for arms
    Task_emb = np.zeros([A, A])
    Task_emb_full = np.zeros([total_samples, total_samples])

    # KMTL with known similarity ------------------------------------- Perform as known global similarity
    # KMTL with known similarity ------------------------------------- Perform as known global similarity
    if algorithm_flag == 'KTL-UCB-TaskSim':
        if DataXY['data_flag'] == 5 or DataXY['data_flag'] == 7 or DataXY['data_flag'] == 77:
            Task_sim = DataXY['ArmKernelSimilarityMatrix']
        else:
            real_arm_features = DataXY['armContext']
            Task_sim = Kern.rbf_kernel(real_arm_features, real_arm_features, bw_prod)

    # KMTL estimate ARM similarity on the fly -------------------------------
    elif algorithm_flag == 'KTL-UCB-TaskSimEst':
        Task_sim = np.zeros([A, A])

        rr = 0  # pointer 1
        # Construct task embeddings ----- A x A
        for i in range(0, A):
            # (N + n_(i, t) , 2)
            Xi = X_total[rr:rr + samples_per_task[i], :]
            cc = 0  # pointer 2
            # Calculate task embedding for the i_th and j_th arms
            for j in range(0, A):
                key_val = tuple((i, j))

                if i in last_arm or j in last_arm or \
                        affinity_dict[key_val]['Task_emb'] is None:
                    Xj = X_total[cc:cc + samples_per_task[j], :]
                    K_task = Kern.rbf_kernel(Xi, Xj, bw_prob)
                    # Task_emb (A x A) and Task_emb_full (t x t)
                    val = np.mean(K_task)
                    Task_emb[i, j] = val
                    affinity_dict[key_val]['Task_emb'] = np.copy(val)
                else:
                    Task_emb[i, j] = np.copy(affinity_dict[key_val]['Task_emb'])

                # ==============================
                # Similarity -------------------
                """
                if i in last_arm or j in last_arm or \
                        simi_dict[key_val]['Task_emb_full'] is None:
                    Xj = X_total[cc:cc + samples_per_task[j], :]
                    # val = Kern.rbf_kernel(Xi, Xj, bw_simi)
                    val = np.exp(-(2 - 2 * Kern.rbf_kernel(Xi, Xj, bw_prob)) * bw_prod)

                    Task_emb_full[rr:rr + samples_per_task[i], cc:cc + samples_per_task[j]] = val
                    simi_dict[key_val]['Task_emb_full'] = np.copy(val)
                else:
                    Task_emb_full[rr:rr + samples_per_task[i], cc:cc + samples_per_task[j]] = \
                        np.copy(simi_dict[key_val]['Task_emb_full'])
                """
                cc = cc + samples_per_task[j]
            rr = rr + samples_per_task[i]

        np.fill_diagonal(Task_emb_full, 0)

        # Calculate task similarity --- A x A
        rr = 0
        for i in range(0, A):
            cc = 0
            for j in range(0, A):
                key_val = tuple((i, j))

                if i in last_arm or j in last_arm or \
                        affinity_dict[key_val]['Task_sim'] is None:
                    sim = Task_emb[i, i] + Task_emb[j, j] - 2 * Task_emb[i, j]  # 2 - 2 * Task_emb[i, j]
                    val = np.exp(-sim * bw_prod)
                    Task_sim[i, j] = val
                    affinity_dict[key_val]['Task_sim'] = np.copy(val)
                else:
                    Task_sim[i, j] = np.copy(affinity_dict[key_val]['Task_sim'])

                cc = cc + samples_per_task[j]
            rr = rr + samples_per_task[i]

        pos = np.where(Task_sim == np.max(Task_sim))
    elif algorithm_flag == 'Lin-UCB-Ind':
        Task_sim = np.identity(A)  # independent task similarity
    elif algorithm_flag == 'Lin-UCB-Pool':
        Task_sim = np.ones([A, A])  # Pooled task similarity
    # -----------------------------------------------------------------------

    # Calculate final kernel (N*A + t x N*A + t)-----------------------------------------------
    K_sim = np.zeros([total_samples, total_samples])
    if algorithm_flag != 'NEW-KTL-UCB-TaskSimEst':
        rr = 0
        for i in range(0, A):
            Xi = X_total[rr:rr + samples_per_task[i], :]
            cc = 0
            for j in range(0, A):
                key_val = tuple((i, j))

                Xj = X_total[cc:cc + samples_per_task[j], :]
                # K_tilde = k_z * k_x

                if i in last_arm or j in last_arm or \
                        affinity_dict[key_val]['K_sim'] is None:
                    val = Task_sim[i, j] * Kern.rbf_kernel(Xi, Xj, bw_x)
                    K_sim[rr:rr + samples_per_task[i], cc:cc + samples_per_task[j]] = val
                    affinity_dict[key_val]['K_sim'] = np.copy(val)
                else:
                    K_sim[rr:rr + samples_per_task[i], cc:cc + samples_per_task[j]] = \
                        np.copy(affinity_dict[key_val]['K_sim'])

                cc = cc + samples_per_task[j]
            rr = rr + samples_per_task[i]
    # ---------------------------------------------------------------------------------

    # correction term --- eta --- diagonal matrix --- N*A + t
    eta_arm = np.zeros([X_total.shape[0]])
    rr = 0
    for i in range(0, A):
        if algorithm_flag == 'KTL-UCB-TaskSim':
            train_dataset = 'Train_Datasets_KTLUCB' + str(i)
        elif algorithm_flag == 'KTL-UCB-TaskSimEst':
            train_dataset = 'Train_Datasets_KTLEstUCB' + str(i)
        elif algorithm_flag == 'NEW-KTL-UCB-TaskSimEst':
            train_dataset = 'Train_Datasets_NEWKTLEstUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Ind':
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)
        # ---- N x 2
        X = np.copy(DataXY[train_dataset])
        # Kronecker product of two arrays. np.kron([1,10,100], [5,6,7])
        # ===>                             array([  5,   6,   7, ..., 500, 600, 700])
        # 1 / N
        eta_arm[rr:rr + X.shape[0]] = np.kron(1 / X.shape[0], np.ones(X.shape[0]))
        rr = rr + X.shape[0]

    eta = np.diag(eta_arm)
    # Second term of S_a,t
    InvTerm = np.linalg.inv(eta.dot(K_sim) + gamma * np.identity(K_sim.shape[0]))

    DataXY[algorithm_flag + '_TaskSim'] = np.copy(Task_sim)
    DataXY[algorithm_flag + '_TaskEmb'] = np.copy(Task_emb)
    DataXY[algorithm_flag + 'KSim'] = np.copy(K_sim)
    DataXY[algorithm_flag + 'etaArm'] = np.copy(eta_arm)
    DataXY[algorithm_flag + 'InvMat'] = np.copy(InvTerm)

    # Task_sim ======> k_Z (z, z') ---- the kernel on Z

    # K_sim ========== capital K_tilde --- total_samples x total_samples
    #                                  --- N*A x N*A
    return K_sim, eta_arm, Task_sim, DataXY