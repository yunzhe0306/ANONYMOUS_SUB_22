import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import DataCreate.DataCreate as DC
import argparse
import time
from datetime import datetime
import sys
from sklearn.preprocessing import normalize


# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open("./Neural-UCB-logs/logfile_" + dt_string + "_Neural-UCB_.log", "w")
        self.out = stdout
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        pass


sys.stdout = Logger(sys.stdout)


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = dim

        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        # self.init_param_paper()

    def init_param_paper(self):
        fc_1_wieghts_diag = np.random.normal(loc=0, scale=(4 / self.hidden_size),
                                             size=(self.hidden_size // 2, self.input_dim // 2))
        fc_1_weights = np.zeros((self.hidden_size, self.input_dim))
        fc_1_weights[0:self.hidden_size // 2, 0:self.input_dim // 2] = fc_1_wieghts_diag
        fc_1_weights[self.hidden_size // 2:, self.input_dim // 2:] = fc_1_wieghts_diag
        self.fc1.weight = nn.Parameter(torch.from_numpy(fc_1_weights).float())

        fc_2_weights_half = np.random.normal(loc=0, scale=(2 / self.hidden_size), size=(1, self.hidden_size // 2))
        fc_2_weights = np.concatenate((fc_2_weights_half, -1 * fc_2_weights_half), axis=1)
        self.fc2.weight = nn.Parameter(torch.from_numpy(fc_2_weights).float())

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, sample_num=1000):
        self.func = Network(dim, hidden_size=hidden).cuda()
        self.input_dim = dim
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        print("Total param num: ", self.total_param)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        # self.U = lamdba * torch.ones((self.total_param,))
        self.nu = nu
        self.sample_num = sample_num

    def init_context_list(self, init_context, init_rewards, A, long_vector_flag=False):
        for i in range(A):
            context, reward = init_context[i, i, :], init_rewards[i, i]
            if long_vector_flag:
                init_dim = int(self.input_dim / A)
                long_context = np.zeros((1, self.input_dim))
                long_context[0, i * init_dim:(i + 1) * init_dim] = context
                context = long_context
            self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
            self.reward.append(reward)

            # ----
            """
            tensor = torch.from_numpy(context).float().cuda()
            mu = self.func(tensor)

            self.func.zero_grad()
            mu.backward()

            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            self.U += g * g
            """

    def init_kernel_matrix(self, A):
        print("Initializing kernel matrix...")
        for c_i in range(A):
            # Update kernel matrix with initial contexts
            tensor = self.context_list[c_i].cuda()
            tensor = tensor.squeeze(dim=0)
            mu = self.func(tensor)

            self.func.zero_grad()
            mu.backward(retain_graph=True)

            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            self.U += g * g

            # TODO: Change to gradually training
            # Train the model with initial contexts
            optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
            length = len(self.reward)
            index = np.arange(length)
            np.random.shuffle(index)
            cnt = 0
            tot_loss = 0

            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= self.sample_num:
                    print("Avg loss: ", tot_loss / self.sample_num)
                    break

                # if batch_loss / length <= 1e-3:
                #     print("Avg loss: ", tot_loss / cnt)
                #     break

    def select(self, context, t):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for i, fx in enumerate(mu):
            self.func.zero_grad()
            fx.backward(retain_graph=True)

            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)

            # CB square
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            if (t+1) % 20 == 0:
                print("Arm: {}, f_hat: {}, UCB: {}".format(i, fx.item(), sample_r))

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]

        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0

        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= self.sample_num:
                    print("Avg loss: ", tot_loss / self.sample_num)
                    return tot_loss / self.sample_num

            if batch_loss / length <= 1e-3:
                print("Avg loss: ", tot_loss / cnt)
                return batch_loss / length


# Output -> X:(A, A*dim)
def generate_vec(t, context, num_dim, num_arm):
    X = np.zeros((num_arm, num_arm * num_dim))
    for a in range(num_arm):
        X[a, a * num_dim:(a + 1) * num_dim] = context[t, a, :]

    return X


# Generate long vector for contexts
def generate_long_vec_category(category_list, context, init_dim, num_arm):
    row_num = len(category_list)
    this_X = np.zeros([row_num, init_dim * num_arm])
    for i, category in enumerate(category_list):
        arm_index = category    # Category index start from 0
        this_X[i, arm_index * init_dim:(arm_index + 1) * init_dim] = context

    return this_X


if __name__ == '__main__':
    """
    Parameters:
    Offset: 100 + 0.1 + extend + not double
    """

    # TODO: Add initial contexts ----

    parser = argparse.ArgumentParser(description='NeuralUCB')
    # nu value: 0.01 / 0.001 / 0.0001
    parser.add_argument('--nu', type=float, default=0.01, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularization')
    # hidden size:
    parser.add_argument('--hidden', type=int, default=500, help='network hidden size')

    # =======================
    torch.cuda.set_device(0)

    # --------------------------------

    #
    multi_estimators = True
    sample_num = 1000

    # Dataset
    data_flag = 12
    A = 19
    N, N_valid = 1, 10
    T = 4000

    if data_flag == 5:
        data_flag_multiclass = 'real-world_XRMB'
    elif data_flag == 7:
        data_flag_multiclass = 'real-world_MNIST'
    elif data_flag == 77:
        data_flag_multiclass = 'real-world_MNIST_Augmented_Labels'
    elif data_flag == 10:
        data_flag_multiclass = 'real-world_Yelp'
    elif data_flag == 11:
        data_flag_multiclass = 'real-world_Yahoo'
    elif data_flag == 12:
        data_flag_multiclass = 'real-world_MovieLens'
    elif data_flag == 13:
        data_flag_multiclass = 'real-world_LastFM'
    elif data_flag == 100:
        data_flag_multiclass = 'new_synthetic_clusters'
    else:
        data_flag_multiclass = None

    print("Data set: ", data_flag_multiclass)
    print(multi_estimators, sample_num)

    randomSeedsTest = np.array([15486101, 15486511, 15486883, 15487271,
                                15486139, 15486517, 15486893, 15487291,
                                15486157, 15486533, 15486907, 15487309,
                                15486173, 15486557, 15486917, 15487313,
                                15486181, 15486571, 15486929, 15487319,
                                15486193, 15486589, 15486931, 15487331,
                                15486209, 15486649, 15486953, 15487361,
                                15486221, 15486671, 15486967, 15487399,
                                15486227, 15486673, 15486997, 15487403,
                                15486241, 15486703, 15487001, 15487429,
                                15486257, 15486707, 15487007, 15487457,
                                15486259, 15486719, 15487019, 15487469])

    RunNumber = 0
    Main_Program_flag = 1

    # Get the train data. This is just one example assigned to each arm randomly when N = 1 (cold start)
    Basic_DataXY = DC.TrainDataCollect(data_flag, A, N_valid, N, T, randomSeedsTest[RunNumber], RunNumber,
                                       Main_Program_flag)

    # ==================================================================================================================
    if data_flag == 5 or data_flag == 7 or data_flag == 77:
        # ==============================================================================================
        assert A == Basic_DataXY['NoOfArms']

        # Real-world classification tasks -- MNIST
        Features_train = Basic_DataXY['TrainContexts']
        Features_test = Basic_DataXY['TestContexts']
        Labels_train_matrix = Basic_DataXY['TrainLabels']
        Labels_test_matrix = Basic_DataXY['TestLabels']
        arm_descriptor_m = Basic_DataXY['ArmDescriptorMatrix']
        A = Basic_DataXY['NoOfArms']

        input_dim = Features_train.shape[1]

        # -----------------------------------------------------------------------
        X = np.zeros((T, A, input_dim))
        init_X = np.zeros((A * N, A, input_dim))

        # (T, A, d) -----------------------------------------------------------------
        for i in range(T):
            for j in range(A):
                normalized_vec = normalize(Features_test[i, :].reshape(1, -1))
                #
                X[i, j, :] = normalized_vec

        # init contexts
        for i in range(A * N):
            for j in range(A):
                normalized_vec = normalize(Features_train[i, :].reshape(1, -1))
                #
                init_X[i, j, :] = normalized_vec

        # Reward matrix
        rewards_m = np.copy(Labels_test_matrix)
        init_reward_m = np.copy(Labels_train_matrix)

    elif data_flag == 10 or data_flag == 12:
        context_matrix = Basic_DataXY['context_matrix']
        init_context_matrix = Basic_DataXY['initContext']
        category_dict = Basic_DataXY['Category_Dict']
        init_category_dict = Basic_DataXY['init_Category_Dict']

        items_per_step = context_matrix.shape[1]

        A = Basic_DataXY['NoOfArms']

        input_dim = context_matrix.shape[2]

        # -----------------------------------------------------------------------
        X = np.zeros((T, items_per_step, input_dim))
        init_X = np.zeros((A * N, items_per_step, input_dim))

        # (T, A, d) -----------------------------------------------------------------
        for i in range(T):
            for j in range(items_per_step):
                normalized_vec = normalize(context_matrix[i, j, :].reshape(1, -1))
                #
                X[i, j, :] = normalized_vec

        # init contexts
        for i in range(A * N):
            for j in range(items_per_step):
                normalized_vec = normalize(init_context_matrix[i, j, :].reshape(1, -1))
                #
                init_X[i, j, :] = normalized_vec

        #
        rewards_m = np.copy(Basic_DataXY['reward_matrix'])
        init_reward_m = np.copy(Basic_DataXY['init_reward_matrix'])
    else:
        # Other data sets ==============================================================================================
        input_dim = Basic_DataXY['userContext'].shape[1]

        X = np.zeros((T, A, input_dim))
        init_X = np.zeros((A * N, A, input_dim))

        user_matrix = Basic_DataXY['userContext']
        arm_matrix = Basic_DataXY['armContext']

        init_user_matrix = Basic_DataXY['initUserContext']

        #
        for i in range(T):
            for j in range(A):
                normalized_vec = normalize(np.multiply(user_matrix[i, :], arm_matrix[j, :]).reshape(1, -1))

                X[i, j, :] = normalized_vec

        #
        for i in range(A * N):
            for j in range(A):
                normalized_vec = normalize(np.multiply(init_user_matrix[i, :], arm_matrix[j, :]).reshape(1, -1))

                init_X[i, j, :] = normalized_vec

        #
        rewards_m = np.copy(Basic_DataXY['reward_matrix'])
        init_reward_m = np.copy(Basic_DataXY['init_reward_matrix'])

    # ==================================================================================================================
    # --------------------------------------------------------------- Neural-UCB starts here

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    # ------------------------------------------
    algorithm_flag = 'Neural-UCB'
    args = parser.parse_args()
    print("nu value: ", args.nu)

    if multi_estimators:
        matrix_dim = input_dim * A
    else:
        matrix_dim = input_dim

    #
    n_UCB = NeuralUCBDiag(matrix_dim, args.lamdba, args.nu, hidden=args.hidden, sample_num=sample_num)
    n_UCB.init_context_list(init_context=init_X, init_rewards=init_reward_m, A=A, long_vector_flag=multi_estimators)

    #
    # n_UCB.init_kernel_matrix(A=A)

    start_time = time.time()
    regrets = []
    summ = 0
    s_count = 0
    for t in range(T):
        # get context and rewards
        if multi_estimators:
            if data_flag == 10 or data_flag == 12:
                # print("Get new long embedded contexts categories...")
                context = np.empty([0, input_dim * A])
                rwd_list = []
                item_pool_s = X.shape[1]
                for a_i in range(item_pool_s):
                    normalized_vec = X[t, a_i, :]
                    category_list = Basic_DataXY['Category_Dict'][tuple([t, a_i])]
                    this_contexts = generate_long_vec_category(category_list, normalized_vec, input_dim, A)
                    context = np.concatenate([context, this_contexts], axis=0)
                    rwd_list += [float(rewards_m[t, a_i])] * this_contexts.shape[0]
                rwd = np.array(rwd_list).reshape(-1, )
                print("This item pool length: ", context.shape[0])
            else:
                context, rwd = generate_vec(t=t, context=X, num_dim=input_dim, num_arm=A), rewards_m[t, :].reshape((A,))
        else:
            if data_flag == 10 or data_flag == 12:
                # # print("Get new long embedded contexts categories...")
                # context = np.empty([0, input_dim])
                # rwd_list = []
                item_pool_s = X.shape[1]
                # for a_i in range(item_pool_s):
                #     normalized_vec = X[t, a_i, :].reshape(1, -1)
                #     category_list_len = len(Basic_DataXY['Category_Dict'][tuple([t, a_i])])
                #     this_contexts = np.repeat(normalized_vec, category_list_len, axis=0)
                #     context = np.concatenate([context, this_contexts], axis=0)
                #     rwd_list += [float(rewards_m[t, a_i])] * this_contexts.shape[0]
                # rwd = np.array(rwd_list).reshape(-1, )
                #
                context, rwd = X[t, :, :].reshape((item_pool_s, input_dim)), rewards_m[t, :].reshape((item_pool_s,))
                print("This item pool length: ", context.shape[0])
            else:
                context, rwd = X[t, :, :].reshape((A, input_dim)), rewards_m[t, :].reshape((A,))

        # Select arm
        arm_select, nrm, sig, ave_rwd = n_UCB.select(context, t)
        r = rwd[arm_select]
        best_arm = np.argwhere(rwd == np.max(rwd)).flatten()
        if arm_select in best_arm:
            s_count += 1
        reg = np.max(rwd) - r
        summ += reg
        if t < 5000:
            loss = n_UCB.train(context[arm_select], r)
        else:
            if t % 100 == 0:
                loss = n_UCB.train(context[arm_select], r)
        regrets.append(summ)

        print("Selected arm: {}, best arm: {}".format(arm_select, best_arm))
        if (t + 1) % 20 == 0:
            print('Time {}: summ: {:.3f}, loss: {:.3e}, nrm: {:.3e}, sig: {:.3e}, ave_rwd: {:.3e}'
                  .format(t + 1, summ, loss, nrm, sig, ave_rwd))
            print("Algorithm: ", algorithm_flag, ", Step: ", t + 1, "/", T, ", Time elapsed: ",
                  time.time() - start_time)
            print("Accuracy of ", algorithm_flag + ": ", str(s_count / t))
