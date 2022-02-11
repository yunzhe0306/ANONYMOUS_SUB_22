import numpy as np
import os
import warnings
import random

from sklearn.cluster import k_means

warnings.filterwarnings("ignore")
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy import stats
from six.moves import cPickle
import sklearn.metrics.pairwise as Kernel

from scipy.io import loadmat
import pickle


def labels_to_reward_matrix(labels, A):
    length = len(labels)
    assert A == (max(labels) + 1)
    reward_m = np.zeros((length, A))
    for i in range(length):
        reward_m[i, int(labels[i])] = 1

    return reward_m


def labels_to_reward_matrix_partial_reward(labels, A, cluster_per_digit, partial_reward_val=0.5):
    length = len(labels)
    assert A == (max(labels) + 1)

    #
    ori_Labels, _ = np.divmod(labels, cluster_per_digit)

    #
    reward_m = np.zeros((length, A))
    for i in range(length):
        reward_m[i, int(ori_Labels[i]) * cluster_per_digit: (int(ori_Labels[i]) + 1) * cluster_per_digit] = \
            partial_reward_val
        reward_m[i, int(labels[i])] = 1

    return reward_m


def get_arm_descriptor_from_contexts_samples(A, context_matrix, labels, sample_num=10):
    d = context_matrix.shape[1]
    descriptor_m = np.zeros([A, d])
    for aa in range(0, A):
        idx = np.where(labels == aa)[0]

        avg_descriptor = np.mean(context_matrix[idx[:sample_num], :], axis=0)
        descriptor_m[aa, :] = avg_descriptor
    return descriptor_m


def get_kernel_embedding_similarity_from_contexts_samples(A, context_matrix, labels, bandwidth, sample_num=350):
    d = context_matrix.shape[1]
    kernel_embedding_m = np.zeros([A, A])

    bw_emb = 1
    for a_i in range(0, A):
        idx_i = np.where(labels == a_i)[0][:sample_num]
        for a_j in range(0, A):
            idx_j = np.where(labels == a_j)[0][:sample_num]

            val = np.mean(Kernel.rbf_kernel(
                context_matrix[idx_i, :].reshape(sample_num, d),
                context_matrix[idx_j, :].reshape(sample_num, d),
                bw_emb
            ))
            kernel_embedding_m[a_i, a_j] = val

    # 20
    bw_simi = 10
    arm_similarity_m = np.zeros([A, A])
    for a_i in range(0, A):
        for a_j in range(0, A):
            sim = kernel_embedding_m[a_i, a_i] + kernel_embedding_m[a_j, a_j] \
                  - 2 * kernel_embedding_m[a_i, a_j]
            val = np.exp(-sim * bw_simi)
            arm_similarity_m[a_i, a_j] = val

    print("Similarity BW: ", bw_emb, bw_simi)
    print("Similarity matrix: ", arm_similarity_m)

    return arm_similarity_m


# ============================================================ Generating training data
def TrainDataCollect(data_flag, A, N_valid, N, T, RandomSeedNumber, RunNumber, Main_Program_flag,
                     READ_PREVIOUS=True, MULTI_RUNS=False, bandwidth=None):
    """
    :param data_flag: Synthetic / Multi-class
    :param data_flag_multiclass: Which multi-class dataset to use
    :param A: Number of arms, only valid for synthetic data
    :param d: Dimension for synthetic data
    :param N_valid: Number of data points per arm in validation set
    :param N: Algorithm starts with one random example assigned to each arm, N = 1
    :param T: Run the UCB for T times.
    :param RandomSeedNumber: Seed for the ramdom number
    :param Main_Program_flag: Equal to 0, do not change
    :return:
    """

    # ------------------------------------------------- Multi-class data ===============================================
    if data_flag == 5:
        # Original XRMB data set (10 arms / nodes)
        rng = np.random.RandomState(RandomSeedNumber)

        #
        mat_1 = loadmat('./Dataset/XRMBf2KALDI_window7_single1.mat')
        mat_2 = loadmat('./Dataset/XRMBf2KALDI_window7_single2.mat')
        Features = mat_1['X1']
        Features = normalize(Features, axis=1)
        Labels = mat_2['trainLabel'].reshape(-1, )

        # Get rid of the last two classes due to insufficient data
        pre_idx = np.where(Labels < 38)[0]
        Features = Features[pre_idx, :]
        Labels = Labels[pre_idx]

        # --------
        assert A == np.unique(Labels).shape[0]
        A = np.unique(Labels).shape[0]  # number of classes
        Features_valid, Features_train_test, Labels_valid, Labels_train_test = train_test_split(Features, Labels,
                                                                                                train_size=int(
                                                                                                    A * N_valid),
                                                                                                random_state=3,
                                                                                                stratify=Labels)
        Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features_train_test,
                                                                                    Labels_train_test,
                                                                                    train_size=int(
                                                                                        A * N),
                                                                                    random_state=RandomSeedNumber,
                                                                                    stratify=Labels_train_test)

        # Sample some labels to obtain arm similarity
        arm_descriptor_m = get_arm_descriptor_from_contexts_samples(A, Features_test, Labels_test)
        arm_kernel_similarity_m = get_kernel_embedding_similarity_from_contexts_samples(A, Features_test, Labels_test,
                                                                                        bandwidth)

        # Features_train.shape, Features_test.shape, A*N, Labels_train
        # A * N --- Training data set
        idx = np.arange(A * N)
        # Permutation of the train dataset
        Features_train = Features_train[idx, :]
        Labels_train = Labels_train[idx]

        # Testing data set
        Labels_test_hist = np.histogram(Labels_test, A)
        # The class with minimal number of instances
        minimum_number_label = np.min(Labels_test_hist[0])
        Features_test_dummy = np.zeros([A * minimum_number_label, Features_test.shape[1]])
        Labels_test_dummy = np.zeros([A * minimum_number_label])

        # Sample 'minimum_number_label' from each class
        for aa in range(0, A):
            idx = np.where(Labels_test == aa)[0]

            Features_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label, :] \
                = Features_test[idx[:minimum_number_label], :]

            Labels_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label] \
                = Labels_test[idx[:minimum_number_label]]

        # Reshuffle the data set
        idx = rng.permutation(A * minimum_number_label)
        Features_test = Features_test_dummy[idx, :]
        Labels_test = Labels_test_dummy[idx]

        # Label matrices
        Labels_train_matrix = labels_to_reward_matrix(Labels_train, A)
        Labels_test_matrix = labels_to_reward_matrix(Labels_test, A)
        print(A * minimum_number_label)

        # ----------------------------s
        DataXY = dict()
        for aa in range(0, A):
            XTrain = Features_train[aa * N: (aa + 1) * N, :]
            LabelsTrain = Labels_train[aa * N: (aa + 1) * N]
            YTrain = np.zeros([N])
            YTrain[LabelsTrain == aa] = 1
            print(LabelsTrain, aa, YTrain)
            Total_Features = np.copy(XTrain)
            Arm_rewards = np.copy(YTrain)

            # Save training data for NEW KTLEst UCB ------------------------------------
            train_datasetNEWKTLEstUCB = 'Train_Datasets_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_datasetNEWKTLEstUCB] = np.copy(Total_Features)

            train_labelsNEWKTLEstUCB = 'Train_Labels_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_labelsNEWKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        DataXY['Testfeatures'] = np.copy(Features_test)
        DataXY['theta'] = 0
        DataXY['armTest'] = np.copy(Labels_test)
        DataXY['d'] = Features_test.shape[1]
        print("Shape: ", Features_train.shape)

        #
        DataXY['TrainContexts'] = Features_train
        DataXY['TestContexts'] = Features_test
        DataXY['TrainLabels'] = Labels_train_matrix
        DataXY['TestLabels'] = Labels_test_matrix

        DataXY['ArmDescriptorMatrix'] = arm_descriptor_m
        DataXY['ArmKernelSimilarityMatrix'] = arm_kernel_similarity_m

        # Number of arms ----------------
        DataXY['NoOfArms'] = A

    elif data_flag == 77:
        # Fine-grained labels for MNIST data set (digit_cluster * 10 classes / nodes)
        rng = np.random.RandomState(RandomSeedNumber)
        # ----------------------
        cluster_per_digit = 5
        print("Cluster per digit: ", cluster_per_digit)

        #
        mnist = datasets.load_svmlight_file('./Dataset/mnist.scale.bz2')
        Features = mnist[0]
        Features = np.array(Features.todense())
        Features = normalize(Features, axis=1)
        ori_Labels = mnist[1]

        # ========================= Label Augmentation
        Labels = np.copy(ori_Labels)
        init_class_num = np.unique(Labels).shape[0]
        for class_i in range(init_class_num):
            this_indices = np.where(ori_Labels == class_i)[0]
            this_features = Features[this_indices, :]

            _, labels, _ = k_means(
                this_features, cluster_per_digit, n_init=50, random_state=RandomSeedNumber, verbose=False
            )

            #
            Labels[this_indices] = (Labels[this_indices] * cluster_per_digit) + labels
            print(np.unique(Labels[this_indices], return_counts=True))

        # =========================

        #
        assert A == (init_class_num * cluster_per_digit)
        A = np.unique(Labels).shape[0]  # number of classes
        Features_valid, Features_train_test, Labels_valid, Labels_train_test = train_test_split(Features, Labels,
                                                                                                train_size=int(
                                                                                                    A * N_valid),
                                                                                                random_state=3,
                                                                                                stratify=Labels)
        Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features_train_test,
                                                                                    Labels_train_test,
                                                                                    train_size=int(
                                                                                        A * N),
                                                                                    random_state=RandomSeedNumber,
                                                                                    stratify=Labels_train_test)

        # Sample some labels to obtain arm similarity
        print("Overall: ", np.unique(Labels_test, return_counts=True))
        arm_descriptor_m = get_arm_descriptor_from_contexts_samples(A, Features_test, Labels_test)
        arm_kernel_similarity_m = get_kernel_embedding_similarity_from_contexts_samples(A, Features_test, Labels_test,
                                                                                        bandwidth)

        # A * N --- Training data set
        idx = np.arange(A * N)
        # Permutation of the train dataset
        Features_train = Features_train[idx, :]
        Labels_train = Labels_train[idx]

        # Testing data set
        Labels_test_hist = np.histogram(Labels_test, A)
        # The class with minimal number of instances
        minimum_number_label = np.min(Labels_test_hist[0])
        Features_test_dummy = np.zeros([A * minimum_number_label, Features_test.shape[1]])
        Labels_test_dummy = np.zeros([A * minimum_number_label])

        # Sample 'minimum_number_label' from each class
        for aa in range(0, A):
            idx = np.where(Labels_test == aa)[0]

            Features_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label, :] \
                = Features_test[idx[:minimum_number_label], :]

            Labels_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label] \
                = Labels_test[idx[:minimum_number_label]]

        # Reshuffle the data set
        idx = rng.permutation(A * minimum_number_label)
        Features_test = Features_test_dummy[idx, :]
        Labels_test = Labels_test_dummy[idx]

        # Label matrices
        partial_reward_val = 0.5
        Labels_train_matrix = labels_to_reward_matrix_partial_reward(Labels_train, A, cluster_per_digit,
                                                                     partial_reward_val=partial_reward_val)
        Labels_test_matrix = labels_to_reward_matrix_partial_reward(Labels_test, A, cluster_per_digit,
                                                                    partial_reward_val=partial_reward_val)
        print(A * minimum_number_label, partial_reward_val)

        # ----------------------------s
        DataXY = dict()
        for aa in range(0, A):
            XTrain = Features_train[aa * N: (aa + 1) * N, :]
            LabelsTrain = Labels_train[aa * N: (aa + 1) * N]
            YTrain = np.zeros([N])
            YTrain[LabelsTrain == aa] = 1
            print(LabelsTrain, aa, YTrain)
            Total_Features = np.copy(XTrain)
            Arm_rewards = np.copy(YTrain)

            # Save training data for NEW KTLEst UCB ------------------------------------
            train_datasetNEWKTLEstUCB = 'Train_Datasets_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_datasetNEWKTLEstUCB] = np.copy(Total_Features)

            train_labelsNEWKTLEstUCB = 'Train_Labels_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_labelsNEWKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        DataXY['Testfeatures'] = np.copy(Features_test)
        DataXY['theta'] = 0
        DataXY['armTest'] = np.copy(Labels_test)
        DataXY['d'] = Features_test.shape[1]
        print("Shape: ", Features_train.shape)

        #
        DataXY['TrainContexts'] = Features_train
        DataXY['TestContexts'] = Features_test
        DataXY['TrainLabels'] = Labels_train_matrix
        DataXY['TestLabels'] = Labels_test_matrix

        DataXY['ArmDescriptorMatrix'] = arm_descriptor_m
        DataXY['ArmKernelSimilarityMatrix'] = arm_kernel_similarity_m

        # Number of arms ----------------
        DataXY['NoOfArms'] = A
    # ==================================================================================================================
    # ==================================================================================================================
    # Real-world Yelp data
    elif data_flag == 10:
        assert A == 20
        context_matrix = np.load('./Matrices_Category/Yelp_context_matrix.npy')
        reward_matrix = np.load('./Matrices_Category/Yelp_reward_matrix.npy')
        with open('./Matrices_Category/Yelp_category_dict.pickle', 'rb') as pk_file:
            category_dict = pickle.load(pk_file)
        similarity_m = np.load('./Matrices_Category/Yelp_category_simi_matrix.npy')
        items_per_step = context_matrix.shape[1]

        DataXY = dict()
        # For each arm, generate training samples by rotating the ellipse
        for aa in range(0, A):
            combined_embedding = context_matrix[aa, aa, :].reshape(1, -1)
            combined_embedding = normalize(combined_embedding, axis=1)

            # X_train -- N x 2 / Y_train --- N x 1
            XTrain = combined_embedding
            YTrain = np.copy(reward_matrix[aa, aa]).reshape(1, 1)

            # one sample +  one label for N = 1
            Total_Features = np.copy(XTrain)  # --- N x 2
            Arm_rewards = np.copy(YTrain).reshape(-1, )  # --- N x 1

            # Save training data for NEW-KTLEst UCB
            train_datasetNEWKTLEstUCB = 'Train_Datasets_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_datasetNEWKTLEstUCB] = np.copy(Total_Features)

            train_labelsNEWKTLEstUCB = 'Train_Labels_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_labelsNEWKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        #
        init_category_dict = {}
        shifted_category_dict = {}
        for i in range(A):
            for j in range(items_per_step):
                init_category_dict[tuple([i, j])] = category_dict.pop(tuple([i, j]))
        for i in range(A, context_matrix.shape[0]):
            for j in range(items_per_step):
                shifted_category_dict[tuple([i - A, j])] = category_dict.pop(tuple([i, j]))
        category_dict = shifted_category_dict

        # User context and arm context ---
        DataXY['context_matrix'] = context_matrix[A * N:, :, :]
        DataXY['reward_matrix'] = reward_matrix[A * N:, :]

        DataXY['initContext'] = context_matrix[:A * N, :, :]
        DataXY['init_reward_matrix'] = reward_matrix[:A * N, :]

        DataXY['theta'] = 0
        DataXY['init_Category_Dict'] = init_category_dict
        DataXY['Category_Dict'] = category_dict
        DataXY['genre_similarity'] = similarity_m
        DataXY['d'] = int(context_matrix.shape[2])
        print("d: ", DataXY['d'])
        DataXY['item_pool_size'] = context_matrix.shape[1]
        DataXY['NoOfArms'] = A
    # ==================================================================================================================
    # ==================================================================================================================
    # Real-world MovieLens data
    if data_flag == 12:
        assert A == 19
        context_matrix = np.load('./Matrices_Category/MovieLens_context_matrix.npy')
        reward_matrix = np.load('./Matrices_Category/MovieLens_reward_matrix.npy')
        with open('./Matrices_Category/MovieLens_category_dict.pickle', 'rb') as pk_file:
            category_dict = pickle.load(pk_file)
        similarity_m = np.load('./Matrices_Category/MovieLens_category_simi_matrix.npy')
        items_per_step = context_matrix.shape[1]

        DataXY = dict()
        # For each arm, generate training samples by rotating the ellipse
        for aa in range(0, A):
            combined_embedding = context_matrix[aa, aa, :].reshape(1, -1)
            combined_embedding = normalize(combined_embedding, axis=1)

            # X_train -- N x 2 / Y_train --- N x 1
            XTrain = combined_embedding
            YTrain = np.copy(reward_matrix[aa, aa]).reshape(1, 1)

            # one sample +  one label for N = 1
            Total_Features = np.copy(XTrain)  # --- N x 2
            Arm_rewards = np.copy(YTrain).reshape(-1, )  # --- N x 1

            # Save training data for NEW-KTLEst UCB
            train_datasetNEWKTLEstUCB = 'Train_Datasets_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_datasetNEWKTLEstUCB] = np.copy(Total_Features)

            train_labelsNEWKTLEstUCB = 'Train_Labels_NEWKTLEstUCB' + str(int(aa))
            DataXY[train_labelsNEWKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)

            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        #
        init_category_dict = {}
        shifted_category_dict = {}
        for i in range(A):
            for j in range(items_per_step):
                init_category_dict[tuple([i, j])] = category_dict.pop(tuple([i, j]))
        for i in range(A, context_matrix.shape[0]):
            for j in range(items_per_step):
                shifted_category_dict[tuple([i - A, j])] = category_dict.pop(tuple([i, j]))
        category_dict = shifted_category_dict

        # User context and arm context ---
        DataXY['context_matrix'] = context_matrix[A * N:, :, :]
        DataXY['reward_matrix'] = reward_matrix[A * N:, :]

        DataXY['initContext'] = context_matrix[:A * N, :, :]
        DataXY['init_reward_matrix'] = reward_matrix[:A * N, :]

        DataXY['theta'] = 0
        DataXY['init_Category_Dict'] = init_category_dict
        DataXY['Category_Dict'] = category_dict
        DataXY['genre_similarity'] = similarity_m
        DataXY['d'] = int(context_matrix.shape[2])
        DataXY['item_pool_size'] = context_matrix.shape[1]
        print("d: ", DataXY['d'])
        DataXY['NoOfArms'] = A

    DataXY['data_flag'] = data_flag
    return DataXY


# ===========================================================

def AllDataCollect(DataXY, algorithm_flag):
    # Get total samples and samples in each dataset
    A = DataXY['NoOfArms']
    total_samples = 0
    # -- Modified from np.zeros([A, 1]) to np.zeros(A)
    samples_per_task = np.zeros(A)
    # print("Sample: ", samples_per_task)
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
        # print DataXY.keys()
        X = np.copy(DataXY[train_dataset])  # -- N x 2
        total_samples = total_samples + X.shape[0]  # samples_per_task * num_arms
        samples_per_task[i] = X.shape[0]  # --- CHANGING with the algorithm

    # Collect all labels and all features
    y = np.zeros(total_samples)
    X_total = np.zeros([total_samples, X.shape[1]])
    rr = 0
    for i in range(0, A):
        if algorithm_flag == 'KTL-UCB-TaskSim':
            train_labels = 'Train_Labels_KTLUCB' + str(i)
            train_dataset = 'Train_Datasets_KTLUCB' + str(i)
        elif algorithm_flag == 'KTL-UCB-TaskSimEst':
            train_labels = 'Train_Labels_KTLEstUCB' + str(i)
            train_dataset = 'Train_Datasets_KTLEstUCB' + str(i)
        elif algorithm_flag == 'NEW-KTL-UCB-TaskSimEst':
            train_labels = 'Train_Labels_NEWKTLEstUCB' + str(i)
            train_dataset = 'Train_Datasets_NEWKTLEstUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Ind':
            train_labels = 'Train_Labels_LinUCB' + str(i)
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_labels = 'Train_Labels_PoolUCB' + str(i)
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)

        # Arm rewards ---
        labels = np.copy(DataXY[train_labels])
        y[rr:rr + labels.shape[0]] = np.copy(DataXY[train_labels])
        X_total[rr:rr + labels.shape[0], :] = np.copy(DataXY[train_dataset])
        rr = rr + labels.shape[0]

    # total_samples -> samples_per_task * num_arms --- N * A
    # samples_per_task -> samples_per_arm,
    # y -> labels, (N*A, 1)
    # X_total -> total_dataset ---- (N*A, 2)

    return total_samples, samples_per_task, y, X_total


def AddData(DataXY, arm_tt, algorithm_flag, X_test, reward_test, tt):
    if algorithm_flag == 'KTL-UCB-TaskSim':
        train_labels = 'Train_Labels_KTLUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_KTLUCB' + str(arm_tt)
        test_label = 'Test_Labels_KTLUCB'
        last_roundXTest = 'Test_Datasets_KTLUCB'
    elif algorithm_flag == 'KTL-UCB-TaskSimEst':
        train_labels = 'Train_Labels_KTLEstUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_KTLEstUCB' + str(arm_tt)
        test_label = 'Test_Labels_KTLEstUCB'
        last_roundXTest = 'Test_Datasets_KTLEstUCB'
    elif algorithm_flag == 'NEW-KTL-UCB-TaskSimEst':
        train_labels = 'Train_Labels_NEWKTLEstUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_NEWKTLEstUCB' + str(arm_tt)
        test_label = 'Test_Labels_NEWKTLEstUCB'
        last_roundXTest = 'Test_Datasets_NEWKTLEstUCB'
    elif algorithm_flag == 'Lin-UCB-Ind':
        train_labels = 'Train_Labels_LinUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_LinUCB' + str(arm_tt)
        test_label = 'Test_Labels_LinUCB'
        last_roundXTest = 'Test_Datasets_LinUCB'
    elif algorithm_flag == 'Lin-UCB-Pool':
        train_labels = 'Train_Labels_PoolUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_PoolUCB' + str(arm_tt)
        test_label = 'Test_Labels_PoolUCB'
        last_roundXTest = 'Test_Datasets_PoolUCB'

    Total_Features = np.copy(DataXY[train_dataset])
    Arm_rewards = np.copy(DataXY[train_labels])

    Total_Features = np.append(Total_Features, X_test, axis=0)
    reward_test = np.ones([1]) * reward_test
    Arm_rewards = np.append(Arm_rewards, reward_test, axis=0)

    DataXY[train_dataset] = np.copy(Total_Features)
    DataXY[train_labels] = np.copy(Arm_rewards)
    DataXY[last_roundXTest] = np.copy(X_test)

    if tt == 0:
        armSelectedTT = np.ones([1]) * arm_tt  # np.empty([0])
    else:
        armSelectedTT = np.copy(DataXY[test_label])
        armSelectedTT = np.append(armSelectedTT, np.ones([1]) * arm_tt, axis=0)
    armSelectedTT = armSelectedTT.astype(int)

    DataXY[test_label] = np.copy(armSelectedTT)

    return DataXY
