from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import BanditModels.Unified_Bandit as Bandits
import time
from scipy import special

# CLUB, SCLUB, K_SCLUB
CLUSTERING_ALGORITHM = 'K_Graph'


# This function runs a particular bandit algorithm for T steps
def ContextBanditUCBRunForTSteps(DataXY, T, data_flag,
                                 bw_x, bw_prob, bw_prod, gamma, alpha,
                                 algorithm_flag, RunNumber, activate_clustering=False):
    """
    :param DataXY: Training data
    :param T: Run the UCB for T times.
    :param data_flag: Synthetic / Multi-class (3 / 4 / 7)
    :param bw_x:        -> Unified_Bandits
    :param bw_prob:     -> Unified_Bandits
    :param bw_prod:     -> Unified_Bandits
    :param gamma:       -> Unified_Bandits
    :param alpha:       -> Unified_Bandits
    :param algorithm_flag: Which algorithm is using.
    :return:
    """

    # # The container for arms clustering ============
    # if CLUSTERING_ALGORITHM == 'K_Graph':
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      # dual: 0.20 -- 5, 0.25 -- 3, 0.15 -- 2
    #                                                                      alpha_theta=0.1,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=5e2,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e4,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.20,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #
    #     """
    #     # MovieLens
    #     # Parameter_Dict[algorithm_flag] = np.array([2.15, 2.15, 100, 0.001, alpha])
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      # dual --
    #                                                                      alpha_theta=0.020,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=2.15,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e5,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=2e2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.20,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #     """
    #
    #     """
    #     # LastFM
    #     # Parameter_Dict[algorithm_flag] = np.array([0.2, 2.15, 100, 0.001, alpha])
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      alpha_theta=0.0005,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=0.46,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e5,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=3e2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.02,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #     """
    #
    #     """
    #     # --------- Yahoo
    #     # Parameter_Dict[algorithm_flag] = np.array([2.15, 2.15, 100, 0.001, alpha])
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      alpha_theta=0.01,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=2.15,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e5,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=3e2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.20,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #     """
    #
    #     """
    #     # ---------- Yelp
    #     # Parameter_Dict[algorithm_flag] = np.array([2.15, 2.15, 100, 0.001, alpha])
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      alpha_theta=0.035,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=2.15,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e5,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=2e2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.20,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #     """
    #
    #     """
    #     # Offsets different clusters ----------
    #     # Parameter_Dict[algorithm_flag] = np.array([2.15, 2.15, 100, 0.001, alpha])
    #     arm_clustering_container = AC.Arm_Cluster_Container_Kernel_Graph(DataXY['NoOfArms'], DataXY['d'], DataXY['N'],
    #                                                                      T, gamma,
    #                                                                      data_flag=data_flag,
    #                                                                      arm_theta=DataXY['theta'],
    #                                                                      # Cluster splitting parameter
    #                                                                      # dual: 0.20 -- 5, 0.25 -- 3, 0.15 -- 2
    #                                                                      alpha_theta=0.035,
    #                                                                      # Bandwidth for reward estimation -- same to
    #                                                                      # other benchmarks
    #                                                                      bw_x=bw_x,
    #                                                                      # Feature similarity bandwidth
    #                                                                      bw_W_x_emb=5e2,
    #                                                                      # Reward similarity bandwidth
    #                                                                      bw_W_y_emb=1e4,
    #                                                                      # Final similarity matrix bandwidth
    #                                                                      bw_final_emb=2,
    #                                                                      # Balance parameter
    #                                                                      Inv_parameter=0.20,
    #                                                                      # Arm distance weight vector
    #                                                                      arm_weight_bw_x=1e2,
    #                                                                      activate_clustering=True)
    #     """
    # else:
    #     arm_clustering_container = None
    # ----------------------

    accuracy_UCB = 0.0
    # rewardAccu = np.zeros([T])
    regretUCB = np.zeros([T])
    Bandit_Algo = Bandits.Unified_Bandit(bw_x, bw_prod, gamma, bw_prob, alpha, data_flag,
                                         algorithm_flag)

    if data_flag == 3:
        Bandit_Algo.set_All_User_Context(DataXY['UserContext'])
        Bandit_Algo.set_All_Arm_Context(DataXY['ArmContext'])
        Bandit_Algo.set_Reward_Distribution(DataXY['rewardDist'])

    # Initialize the lists -------------------------------------------------------------------------
    # if arm_clustering_container.activate_clustering and \
    #         (algorithm_flag == 'NEW-KTL-UCB-TaskSimEst'):
    #     for i in range(DataXY['NoOfArms']):
    #         train_datasetKTLEstUCB = 'Train_Datasets_NEWKTLEstUCB' + str(int(i))
    #         arm_context = DataXY[train_datasetKTLEstUCB].reshape(1, -1)
    #
    #         train_labelsKTLEstUCB = 'Train_Labels_NEWKTLEstUCB' + str(int(i))
    #         arm_rewards = DataXY[train_labelsKTLEstUCB]
    #
    #         input_X_Test = arm_context
    #         # input_X_Test = special.expit(arm_context)
    #
    #         if arm_context.shape[1] > DataXY['d']:
    #             num = int(arm_context.shape[1] / int(DataXY['d']))
    #             arm_context = arm_context.reshape((-1, int(DataXY['d'])))
    #             arm_rewards = arm_rewards.reshape((-1, 1))
    #             for index in range(num):
    #                 context, reward = arm_context[index], arm_rewards[index]
    #                 if CLUSTERING_ALGORITHM == 'K_Graph':
    #                     arm_clustering_container.update_user_context(arm_index=i, user_context=[context],
    #                                                                  y_t=[reward], tt=-1)
    #                     arm_clustering_container.update_arm_selected(i)
    #                     # arm_clustering_container.update_clusters(tt=-1, arm_index=i)
    #         else:
    #             # Update context for kernel SCLUB
    #             if CLUSTERING_ALGORITHM == 'K_Graph':
    #                 arm_clustering_container.update_user_context(arm_index=i, user_context=arm_context,
    #                                                              y_t=[arm_rewards], tt=-1)
    #                 arm_clustering_container.update_arm_selected(i)
    #                 # arm_clustering_container.update_clusters(tt=-1, arm_index=i)
    #
    #     if CLUSTERING_ALGORITHM == 'K_Graph':
    #         arm_clustering_container.initial_graph()
    #     arm_clustering_container.update_clusters_info(tt=-1)
    # -----------------------------------------------------------------------------------------------

    # Initializing the affinity matrix dict
    A = DataXY['NoOfArms']
    affinity_dict = dict()
    for i in range(A):
        for j in range(A):
            sub_dict = {'Task_emb': None, 'Task_sim': None, 'K_sim': None, 'K_x': None}
            affinity_dict[tuple((i, j))] = sub_dict
        sub_dict = {'Task_emb': None, 'Task_sim': None, 'K_sim': None, 'K_x': None}
        affinity_dict[i] = sub_dict
    DataXY['affinity_dict'] = affinity_dict

    # Initialize the similarity matrix
    simi_dict = dict()
    for i in range(A):
        for j in range(A):
            sub_dict = {'S_y_sim': None, 'S_y_emb': None, 'Task_emb_full': None, 'final_F': None}
            simi_dict[tuple((i, j))] = sub_dict
    DataXY['simi_dict'] = simi_dict

    # ----
    Selected_Arm_T = np.zeros([T])
    DataXY['selected_arm'] = {i: None for i in range(T)}
    Exact_Arm_T = np.zeros([T])
    Task_sim_dict = dict()
    start_time = time.time()

    cum_regret = 0.0
    arm_selected_count = np.zeros([A])

    for tt in range(0, T):
        # arm_tt -> CHOSEN arm of maximum p_(a, t)
        arm_tt, X_test, DataXY, estimated_Rewards, reward_cluster = Bandit_Algo.get_Arm_And_X_test_And_Data(DataXY, tt)

        if data_flag == 5 or data_flag == 7 or data_flag == 77:
            reward_matrix = DataXY['TestLabels']
            rewardVal = reward_matrix[tt, :]
            ind_arms = np.argwhere(rewardVal == np.max(rewardVal)).flatten()
            if int(arm_tt) in ind_arms:
                ind_arm = arm_tt
            else:
                ind_arm = ind_arms[0]

            true_reward = rewardVal[ind_arm]

            Bandit_Algo.update_Collected_Rewards(DataXY, tt, ind_arm)
            rewardAccu = Bandit_Algo.get_Collected_Rewards()

        # For real-world data
        if data_flag == 10 or data_flag == 12:
            pool_size = DataXY['item_pool_size']
            category_dict = DataXY['Category_Dict']
            reward_matrix = DataXY['reward_matrix']
            pool_sample_size = 0
            context_2_arm_list = []
            context_2_item_list = []
            context_2_reward_list = []
            for i in range(pool_size):
                this_reward = reward_matrix[tt, i]
                arm_list_i = category_dict[tuple([tt, i])]
                for arm in arm_list_i:
                    context_2_arm_list.append(arm)
                    context_2_item_list.append(i)
                    context_2_reward_list.append(this_reward)
                    pool_sample_size += 1

            #
            rewardVal = context_2_reward_list
            ind_arms = np.argwhere(rewardVal == np.max(rewardVal)).flatten()
            if int(arm_tt) in ind_arms:
                ind_arm = arm_tt
            else:
                ind_arm = ind_arms[0]

            true_reward = rewardVal[ind_arm]

            Bandit_Algo.update_Collected_Rewards(DataXY, tt, ind_arm)
            rewardAccu = Bandit_Algo.get_Collected_Rewards()

        Selected_Arm_T[tt] = arm_tt
        if data_flag == 10 or data_flag == 12:
            DataXY['selected_arm'][tt] = set(category_dict[tuple([tt, context_2_item_list[arm_tt]])])
        else:
            DataXY['selected_arm'][tt] = [arm_tt]
        Exact_Arm_T[tt] = ind_arm

        # Accurate Recommendation.
        if int(ind_arm) == int(arm_tt):
            accuracy_UCB += 1

        if data_flag == 5:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 7:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 77:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 10:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 11:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 12:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 13:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 99:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 100:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]
        elif data_flag == 101:
            regretUCB[tt] = rewardVal[ind_arm] - rewardAccu[tt]

        cum_regret += regretUCB[tt]

        # Update clustering information
        # ----------------------------------------------------------------------------
        # if arm_clustering_container.activate_clustering and (algorithm_flag == 'NEW-KTL-UCB-TaskSimEst'):
        #
        #     # Update context for kernel SCLUB
        #     if CLUSTERING_ALGORITHM == 'K_Graph':
        #         arm_clustering_container.update_arm_selected(arm_tt)
        #         arm_clustering_container.update_user_context(arm_index=arm_tt, user_context=X_test,
        #                                                      y_t=[rewardAccu[tt]], tt=tt,
        #                                                      running_status=True)
        #
        #     # arm_clustering_container.update_clusters_info(tt=tt)
        #     arm_clustering_container.update_clusters(tt=tt, arm_index=arm_tt)
        # ----------------------------------------------------------------------------

        # Add Data to the container ---- UserContext(XTest), Armrewards,
        if data_flag == 10 or data_flag == 12:
            for arm_i in DataXY['Category_Dict'][tuple([tt, context_2_item_list[arm_tt]])]:
                DataXY = DC.AddData(DataXY, arm_i, algorithm_flag, X_test, rewardAccu[tt], tt)
                arm_selected_count[arm_i] += 1
            print("Time: ", tt, ", selected arm: ", arm_tt)
        else:
            DataXY = DC.AddData(DataXY, arm_tt, algorithm_flag, X_test, rewardAccu[tt], tt)
            arm_selected_count[arm_tt] += 1
            print("Time: ", tt, ", selected arm: ", arm_tt, ", count: ", int(arm_selected_count[arm_tt]))

        if tt % 20 == 0 and tt != 0:
            print("Algorithm: ", algorithm_flag, ", Step: ", tt, "/", T, ", Time elapsed: ", time.time() - start_time)
            print("Accuracy of ", algorithm_flag + ": ", str(accuracy_UCB / tt))
            print("Cumulative regret: ", cum_regret)
            print("Estimated reward: ", estimated_Rewards[arm_tt])

        if tt % 100 == 0 and tt != 0:
            print("=" * 30, "\n")
            print("-------------------Running status--------------------")
            print("RunNumber: ", RunNumber)
            print("iteration number: ", tt + 1, "/", T, " || Algorithm: ", algorithm_flag)
            print("True class: ", int(ind_arm))
            print("UCB class: ", int(arm_tt))
            print("True reward: ", true_reward)
            print("UCB reward: ", rewardAccu[tt])
            # Accuracy of guessing the right arm
            print("! Accuracy of ", algorithm_flag + ": ", str(accuracy_UCB / tt))
            print("! Cumulative regret: ", cum_regret)
            print("K_sim: ", DataXY[algorithm_flag + 'KSim'])
            print("Task_sim: ", DataXY[algorithm_flag + '_TaskSim'])
            print("=" * 30, "\n")

        if tt == (T - 1):
            print("=" * 30, "\n")
            print("-------------------Final status--------------------")
            print("RunNumber: ", RunNumber)
            print("iteration number: ", tt + 1, "/", T, " || Algorithm: ", algorithm_flag)
            print("True class: ", int(ind_arm))
            print("UCB class: ", int(arm_tt))
            print("True reward: ", true_reward)
            print("UCB reward: ", rewardAccu[tt])
            print("Accuracy of ", algorithm_flag + ": ", str(accuracy_UCB / tt))
            print("! Cumulative regret: ", cum_regret)
            print("K_sim: ", DataXY[algorithm_flag + 'KSim'])
            print("Task_sim: ", DataXY[algorithm_flag + '_TaskSim'])
            print("=" * 30, "\n")

        if tt % (T // 4) == 0:
            Task_sim_dict[algorithm_flag + '_Task_Sim_' + str(tt)] = DataXY[algorithm_flag + '_TaskSim']
    AverageRegret = np.sum(regretUCB) / float(T)
    AverageAccuracy = accuracy_UCB / float(T)

    return AverageRegret, AverageAccuracy, regretUCB, Selected_Arm_T, Exact_Arm_T, Task_sim_dict
