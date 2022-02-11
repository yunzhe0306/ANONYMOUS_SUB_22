from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import Algorithms.UCB as UCB
import warnings

# warnings.filterwarnings("ignore")

import networkx as nx


# f_bw <- bw_x_estKTLUCB (bw_x)
# s_bw <- bw_prod_estKTLUCB (bw_prod)
# gamma <- gammaKTLUCB
# t_bw <- bw_prob_estKTLUCB (bw_prod)
class Unified_Bandit:
    def __init__(self, f_bw, s_bw, gamma, t_bw, alpha, data_flag, type):
        # Names mapping: bw_x, bw_prod, gamma, bw_prob, alpha, data_flag, algorithm_flag
        self.type = type  # Algorithm_flag
        self.f_bw = f_bw
        self.s_bw = s_bw
        self.gamma = gamma
        self.t_bw = t_bw
        self.alpha = alpha
        self.data_flag = data_flag
        self.collected_rewards = []

    # Perform the UCB at time-step t--------------------------------------
    def get_Arm_And_X_test_And_Data(self, data, time_Step):
        estimated_Rewards, reward_cluster, X_test, data = UCB.ContextBanditUCB(data, time_Step, self.f_bw,
                                                                               self.s_bw, self.gamma,
                                                               self.t_bw, self.alpha, self.data_flag, self.type)
        self.selected_arm = np.argmax(estimated_Rewards)
        return self.selected_arm, X_test, data, estimated_Rewards, reward_cluster

    # --------------------------------------------------------------------

    # Return r_(a, t) for synthetic data
    # Reward_Distribution --- DataXY['rewardDist'] ----- np.arange(0, A) / A
    def get_User_Context_Based_Reward(self, user_Context, arm):
        rotated_Context = self.get_Rotatated_Context(user_Context, arm)
        return 1.0 - (rotated_Context[:, 1] - self.Reward_Distribution[arm] + 0.5) ** 2

    # Update rewards ----- self.collected_rewards => rewards-------------------------
    def update_Collected_Rewards(self, DataXY, time_Step, ind_arm):
        if self.data_flag == 5:
            reward_matrix = DataXY['TestLabels']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])
        elif self.data_flag == 7:
            reward_matrix = DataXY['TestLabels']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])
        elif self.data_flag == 77:
            reward_matrix = DataXY['TestLabels']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])
        # New real-world data ==================
        elif self.data_flag == 10 or self.data_flag == 12:
            pool_size = DataXY['item_pool_size']
            category_dict = DataXY['Category_Dict']
            reward_matrix = DataXY['reward_matrix']
            pool_sample_size = 0
            context_2_arm_list = []
            context_2_item_list = []
            context_2_reward_list = []
            for i in range(pool_size):
                this_reward = reward_matrix[time_Step, i]
                arm_list_i = category_dict[tuple([time_Step, i])]
                for arm in arm_list_i:
                    context_2_arm_list.append(arm)
                    context_2_item_list.append(i)
                    context_2_reward_list.append(this_reward)
                    pool_sample_size += 1
            #
            rewardVal = context_2_reward_list[self.selected_arm]
            self.collected_rewards.append([rewardVal])
        elif self.data_flag == 11:
            reward_matrix = DataXY['reward_matrix']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])
        elif self.data_flag == 13:
            reward_matrix = DataXY['reward_matrix']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])

        # New synthetic data
        elif self.data_flag == 99:
            reward_matrix = DataXY['reward_matrix']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])

        # New synthetic data with offsets
        elif self.data_flag == 100:
            reward_matrix = DataXY['reward_matrix']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])

        # New synthetic data with sanity check
        elif self.data_flag == 101:
            reward_matrix = DataXY['reward_matrix']
            rewardVal = reward_matrix[time_Step, self.selected_arm]
            self.collected_rewards.append([rewardVal])

    # All these functions are needed for data_flag = 3 ------------------------------
    def get_Reward_For_All_Arms(self, data, time_Step):
        A = data['NoOfArms']
        RewardforAllArms = np.zeros([A])
        user_Context = self.get_User_Context(time_Step)
        for aa in range(0, A):
            RewardforAllArms[aa] = self.get_User_Context_Based_Reward(user_Context, aa)
        return RewardforAllArms

    def set_All_User_Context(self, context):
        self.User_Context = context

    def set_All_Arm_Context(self, context):
        self.Arm_Context = context

    def set_Reward_Distribution(self, rew_Dist):
        self.Reward_Distribution = rew_Dist

    def get_User_Context(self, time_Step):
        # Check the size of user context if it is 0, then raise an exception
        x = self.User_Context[time_Step, :]
        return x[np.newaxis, :]

    def get_Arm_Context(self):
        # Check the size of arm context if it is 0, then raise an exception
        return self.Arm_Context

    def get_Reward_Distribuiton(self):
        return self.Reward_Distribution

    def get_Rotatated_Context(self, user_context, index):
        mat = np.array([[np.cos(self.Arm_Context[index, 0]), -np.sin(self.Arm_Context[index, 0])],
                        [np.sin(self.Arm_Context[index, 0]), np.cos(self.Arm_Context[index, 0])]])
        return mat.dot(user_context.T).T

    def get_Collected_Rewards(self):
        return self.collected_rewards

    # All the following functions are needed for data_flag = 4,5 -------------------
    def set_Reward_Function_Flag(self, reward_funct):
        self.Reward_Funct = reward_funct

    def set_Hier_Graph(self, Graph):
        self.Hier_Graph = Graph

    def set_Hier_Classes(self, Classes):
        self.Hier_Classes = Classes



