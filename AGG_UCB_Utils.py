from sklearn.cluster import SpectralClustering

import networkx as nx
import numpy as np

from sklearn.utils.validation import check_symmetric
import DataCreate.DataCreate as DC
from sklearn.preprocessing import normalize

from sklearn.manifold import *
from sklearn.cluster._kmeans import k_means
from sklearn_extra.cluster import KMedoids

import sklearn.metrics.pairwise as Kernel
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from networkx.algorithms.cuts import conductance

from scipy.sparse import csr_matrix, vstack
from scipy.linalg import block_diag


def get_sym_norm_matrix(source):
    _ = check_symmetric(source, raise_exception=True)
    input_matrix = np.array(source, dtype=np.float)
    D_matrix = np.diag(np.sum(input_matrix, axis=1))
    D_matrix_minus_1_2 = np.zeros(D_matrix.shape)
    np.fill_diagonal(D_matrix_minus_1_2, 1 / (D_matrix.diagonal() ** 0.5))

    S_y = np.matmul(np.matmul(D_matrix_minus_1_2, input_matrix), D_matrix_minus_1_2)
    return S_y


def get_exact_PPR_matrix(sym_norm_adj_m, PPR_alpha=0.1):
    print("PPR alpha: ", PPR_alpha)
    n_node = sym_norm_adj_m.shape[0]
    inv_m = np.linalg.inv(
        np.identity(n_node) - ((1 - PPR_alpha) * sym_norm_adj_m)
    )

    return PPR_alpha * inv_m


# ======================================================================================================================
def get_arm_true_clustering(A, arm_features, threshold, bandwidth, float_weight=False, diag_val=-1, simi_m=None):
    arm_G = nx.empty_graph(A, create_using=nx.Graph())

    if simi_m is None:
        distance_m = Kernel.rbf_kernel(arm_features, arm_features, bandwidth)
    else:
        distance_m = simi_m
    edge_list = np.where(distance_m > threshold)
    edge_list = np.concatenate((edge_list[0].reshape(1, -1), edge_list[1].reshape(1, -1)), axis=0).T.tolist()

    print("Edge list: ", edge_list)

    # Add edges
    arm_G.add_edges_from(edge_list)

    # Float-type edge weights
    if float_weight:
        float_weight_list = []
        for pos in edge_list:
            weight = distance_m[pos[0], pos[1]]
            float_weight_list.append([pos[0], pos[1], weight])
        print(float_weight_list)
        arm_G.add_weighted_edges_from(float_weight_list)

    # new diagonal value
    if diag_val > 0:
        for i in range(A):
            arm_G[i][i]['weight'] = diag_val

    init_adj = nx.adjacency_matrix(arm_G).todense()
    print("Initial true adj matrix: ")
    print(init_adj)
    print("Initial true edges: ", len(arm_G.edges))

    return arm_G


def get_arm_true_clustering_given_affinity_matrix(A, affinity_m, threshold, float_weight=False, diag_val=-1):
    arm_G = nx.empty_graph(A, create_using=nx.Graph())
    distance_m = affinity_m
    edge_list = np.where(distance_m > threshold)
    edge_list = np.concatenate((edge_list[0].reshape(1, -1), edge_list[1].reshape(1, -1)), axis=0).T.tolist()

    print("Edge list: ", edge_list)

    # Add edges
    arm_G.add_edges_from(edge_list)

    # Float-type edge weights
    if float_weight:
        float_weight_list = []
        for pos in edge_list:
            weight = distance_m[pos[0], pos[1]]
            float_weight_list.append([pos[0], pos[1], weight])
        print(float_weight_list)
        arm_G.add_weighted_edges_from(float_weight_list)

    # new diagonal value
    if diag_val > 0:
        for i in range(A):
            arm_G[i][i]['weight'] = diag_val

    init_adj = nx.adjacency_matrix(arm_G).todense()
    print("affinity Initial true adj matrix: ")
    print(init_adj)
    print("affinity Initial true edges: ", len(arm_G.edges))

    return arm_G


# Generate long vector for contexts
def generate_long_vec(arm_index, context, init_dim, num_arm):
    this_X = np.zeros((init_dim * num_arm, ))
    this_X[arm_index * init_dim:(arm_index + 1) * init_dim] = context

    return this_X


# Generate long vector for contexts
def generate_long_vec_category(context, A):
    # (1, d) -> (A, A * d)
    this_context = context.reshape(1, -1)
    diag_matrix = block_diag(*np.repeat(this_context, A, axis=0))

    return diag_matrix


# Generate long vector for contexts --- BASED on clustering
def generate_long_vec_cluster(arm_index, context, init_dim, num_cluster, arm_2_cluster_dict):
    cluster_index = arm_2_cluster_dict[arm_index]

    this_X = np.zeros((init_dim * num_cluster,))
    this_X[cluster_index * init_dim:(cluster_index + 1) * init_dim] = context

    return this_X


# ======================================================================================================================
def get_datasets_and_rewards(A, T, N, N_valid, threshold, bandwidth, cluster_num=20, data_flag=-1,
                             TRUE_CLUSTERING=True, Model_mode='clustering', FLOAT_WEIGHT=False, NEW_SELF_LOOP_VAL=-1):
    #
    arm_2_cluster_dict = None

    #
    if data_flag == 5 or data_flag == 77:
        # For augmented MNIST and XRMB data set
        Basic_DataXY = DC.TrainDataCollect(data_flag, A, N_valid, N, T, 1, RunNumber=0, Main_Program_flag=1,
                                           bandwidth=bandwidth)
        # ==============================================================================================
        assert A == Basic_DataXY['NoOfArms']
        assert Model_mode == 'independent'

        # Real-world classification tasks -- MNIST
        Features_train = Basic_DataXY['TrainContexts']
        Features_test = Basic_DataXY['TestContexts']
        Labels_train_matrix = Basic_DataXY['TrainLabels']
        Labels_test_matrix = Basic_DataXY['TestLabels']
        arm_descriptor_m = Basic_DataXY['ArmDescriptorMatrix']
        arm_kernel_similarity_m = Basic_DataXY['ArmKernelSimilarityMatrix']
        A = Basic_DataXY['NoOfArms']

        input_dim = Features_train.shape[1]
        init_dim = input_dim

        true_graph = get_arm_true_clustering(A, arm_features=arm_descriptor_m, threshold=threshold, bandwidth=bandwidth,
                                             float_weight=FLOAT_WEIGHT, diag_val=NEW_SELF_LOOP_VAL,
                                             simi_m=arm_kernel_similarity_m)
        # -----------------------------------------------------------------------
        if Model_mode == 'independent':
            # CHANGE TO LONG VEC
            input_dim = input_dim * A
            long_vec_dim = init_dim * A
        # -----------------------------------------------------------------------
        if TRUE_CLUSTERING:
            arm_G = true_graph
        else:
            arm_G = None
        # ===================
        # X = np.zeros((T, A, long_vec_dim))
        X_list = []
        X_short = np.zeros((T, A, init_dim))  # Contexts without extension

        # init_X = np.zeros((A * N, A, long_vec_dim))
        init_X_list = []
        init_X_short = np.zeros((A * N, A, init_dim))

        #
        for i in range(T):
            step_X_list = []
            for j in range(A):
                    normalized_vec = normalize(Features_test[i, :].reshape(1, -1))
                    X_short[i, j, :] = normalized_vec
                    #
                    normalized_long_vec = csr_matrix(generate_long_vec(j, normalized_vec, init_dim, A).reshape(1, -1))
                    step_X_list.append(normalized_long_vec)
            step_x = vstack(step_X_list)
            X_list.append(step_x)
        X = X_list

        # init contexts
        for i in range(A * N):
            init_step_X_list = []
            normalized_vec = normalize(Features_train[i, :].reshape(1, -1))
            for j in range(A):
                init_X_short[i, j, :] = normalized_vec
                #
                normalized_long_vec = csr_matrix(generate_long_vec(j, normalized_vec, init_dim, A).reshape(1, -1))
                init_step_X_list.append(normalized_long_vec)
            init_step_x = vstack(init_step_X_list)
            init_X_list.append(init_X_short[i, :, :])
        init_X = init_X_list

        # Reward matrix -----------------------------------------------------------------
        rewards_m = np.copy(Labels_test_matrix)
        init_reward_m = np.copy(Labels_train_matrix)

    # -----------------------------------------------------------------------
    else:
        # Get the train data. This is just one example assigned to each arm randomly when N = 1 (cold start)
        Basic_DataXY = DC.TrainDataCollect(data_flag, A, N_valid, N, T, 1, RunNumber=0, Main_Program_flag=1)
        # Other data sets ==============================================================================================
        context_matrix = Basic_DataXY['context_matrix']
        init_context_matrix = Basic_DataXY['initContext']
        category_dict = Basic_DataXY['Category_Dict']
        init_category_dict = Basic_DataXY['init_Category_Dict']
        arm_kernel_similarity_m = Basic_DataXY['genre_similarity']

        input_dim = Basic_DataXY['d']
        init_dim = input_dim

        # -----------------------------------------------------------------------
        items_per_step = context_matrix.shape[1]
        # Independent settings
        long_vec_dim = init_dim * A
        X = np.zeros((T, items_per_step, long_vec_dim))
        # init_X = np.zeros((A * N, items_per_step, long_vec_dim))
        init_X = []

        X_short = np.zeros((T, items_per_step, init_dim))  # Contexts without extension
        init_X_short = np.zeros((A * N, items_per_step, init_dim))

        # ------------- Initialize true arm graph
        # true_graph = get_arm_true_clustering(A, arm_features=arm_matrix, threshold=threshold, bandwidth=bandwidth,
        #                                      float_weight=FLOAT_WEIGHT, diag_val=NEW_SELF_LOOP_VAL)
        true_graph = get_arm_true_clustering(A, arm_features=None, threshold=threshold, bandwidth=bandwidth,
                                             float_weight=FLOAT_WEIGHT, diag_val=NEW_SELF_LOOP_VAL,
                                             simi_m=arm_kernel_similarity_m)

        #
        if Model_mode == 'independent':
            # CHANGE TO LONG VEC
            input_dim = input_dim * A
        # ------------------------
        if TRUE_CLUSTERING:
            arm_G = true_graph
        else:
            arm_G = None

        # (T, A, d)
        for i in range(T):
            for j in range(items_per_step):
                normalized_vec = normalize(context_matrix[i, j, :].reshape(1, -1))

                X_short[i, j, :] = normalized_vec

        # init contexts
        for i in range(A):
            for j in range(items_per_step):
                normalized_vec = normalize(init_context_matrix[i, j, :].reshape(1, -1))

                init_X_short[i, j, :] = normalized_vec

        init_reward_list = []
        for i in range(A):
            for j in range(items_per_step):
                normalized_vec = init_X_short[i, j, :]
                if i in init_category_dict[tuple([i, j])]:
                    normalized_long_vec = generate_long_vec_category(normalized_vec, A)
                    init_X.append(normalized_long_vec)
                    break

        # (T, A)
        rewards_m = np.copy(Basic_DataXY['reward_matrix'])
        init_reward_m = np.copy(Basic_DataXY['init_reward_matrix'])

    return Basic_DataXY, input_dim, init_dim, arm_G, true_graph, \
        init_X, init_X_short, init_reward_m, cluster_num, X, X_short, rewards_m, arm_2_cluster_dict


# ============================================================================ Arm graph clustering

def get_arm_clustering_sub_graph(A, arm_graph):
    sub_graphs_list = list([c for c in nx.connected_components(arm_graph)])
    num_cluster = len(sub_graphs_list)

    cluster_2_arm_dict = {i: [] for i in range(num_cluster)}
    arm_2_cluster_dict = {j: None for j in range(A)}

    for g_i, sub_graph in enumerate(sub_graphs_list):
        for arm in sub_graph:
            cluster_2_arm_dict[g_i].append(arm)
            arm_2_cluster_dict[arm] = g_i

    return num_cluster, cluster_2_arm_dict, arm_2_cluster_dict


def get_arm_clustering_local_seeds(A, arm_graph, seed_num=10, n_runs=1000):
    best_seed_list, best_seed_2_arm_dict, best_arm_2_seed_dict, best_arm_2_cluster_dict, best_estimator_count, \
    best_isolated_nodes_list = None, None, None, None, None, None
    min_inertia = np.inf

    for run_i in range(n_runs):
        seed_list = sorted(np.random.choice(A, size=(seed_num,), replace=False).tolist())
        path_pairs_dict = dict(nx.all_pairs_shortest_path_length(arm_graph))
        # sub_graphs_list = list([c for c in nx.connected_components(arm_graph)])
        # num_cluster = len(sub_graphs_list)

        #
        seed_2_arm_dict = {i: [] for i in seed_list}
        arm_2_seed_dict = {j: None for j in range(A)}
        arm_2_cluster_dict = {j: None for j in range(A)}
        estimator_count = seed_num
        isolated_nodes_list = []

        for source_n in range(A):
            min_distance, this_seed_node, this_cluster = np.inf, None, None
            for cluster_index, seed in enumerate(seed_list):
                if seed in path_pairs_dict[source_n]:
                    if min_distance > path_pairs_dict[source_n][seed]:
                        this_seed_node = seed
                        this_cluster = cluster_index
                        min_distance = path_pairs_dict[source_n][seed]

            if this_seed_node is not None:
                arm_2_seed_dict[source_n] = this_seed_node
                seed_2_arm_dict[this_seed_node].append(source_n)
                arm_2_cluster_dict[source_n] = this_cluster
            else:
                estimator_count += 1
                isolated_nodes_list.append(source_n)

        #
        for i, n_i in enumerate(isolated_nodes_list):
            arm_2_cluster_dict[n_i] = i + seed_num

        # Calculate the inertia
        total_inertia_seed = 0.0

        if total_inertia_seed < min_inertia:
            min_inertia = total_inertia_seed
            best_seed_list, best_seed_2_arm_dict, best_arm_2_seed_dict, best_arm_2_cluster_dict, best_estimator_count, \
            best_isolated_nodes_list \
                = seed_list, seed_2_arm_dict, arm_2_seed_dict, arm_2_cluster_dict, estimator_count, isolated_nodes_list

    # Minimal Inertia seed
    print("Seed list: ", best_seed_list)
    print("Seed num: ", seed_num, len(best_isolated_nodes_list))
    print(best_seed_2_arm_dict)
    print("Runs: ", n_runs, ", Min seed inertia: ", min_inertia)

    best_cluster_2_arm_dict = {j: [] for j in range(seed_num)}
    for a_i in range(A):
        c_i = best_arm_2_cluster_dict[a_i]
        best_cluster_2_arm_dict[c_i].append(a_i)

    return best_seed_list, best_seed_2_arm_dict, best_arm_2_seed_dict, best_arm_2_cluster_dict, best_estimator_count, \
           best_isolated_nodes_list, best_cluster_2_arm_dict, min_inertia


# ================================================================================================

def get_arm_clustering_spectral_clustering(A, arm_features, arm_graph, cluster_num=20):
    adj_matrix = nx.adjacency_matrix(arm_graph).todense()
    sc = SpectralClustering(cluster_num, affinity='precomputed', assign_labels='kmeans', n_init=1000)
    # sc = SpectralClustering(cluster_num, affinity='rbf', assign_labels='kmeans', n_init=1000)

    sc.fit(adj_matrix)
    # sc.fit(arm_features)
    arm_c_labels = sc.labels_
    affinity_m = sc.affinity_matrix_

    #
    cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
    arm_2_cluster_dict = {j: None for j in range(A)}

    for a_i, c_label in enumerate(arm_c_labels):
        cluster_2_arm_dict[c_label].append(a_i)
        arm_2_cluster_dict[a_i] = c_label

    c_labels_hist = np.histogram(arm_c_labels, cluster_num)

    #
    print(c_labels_hist)
    print("Cluster num: ", cluster_num)
    print(cluster_2_arm_dict)

    return cluster_2_arm_dict, arm_2_cluster_dict, arm_c_labels


def get_arm_clustering_spectral_clustering_with_features(A, arm_features, bandwidth, cluster_num=20):
    sc = SpectralClustering(cluster_num, gamma=bandwidth, affinity='rbf', assign_labels='kmeans', n_init=1000)

    sc.fit(arm_features)
    arm_c_labels = sc.labels_
    affinity_m = sc.affinity_matrix_

    #
    cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
    arm_2_cluster_dict = {j: None for j in range(A)}

    for a_i, c_label in enumerate(arm_c_labels):
        cluster_2_arm_dict[c_label].append(a_i)
        arm_2_cluster_dict[a_i] = c_label

    c_labels_hist = np.histogram(arm_c_labels, cluster_num)

    #
    print(c_labels_hist)
    print("Cluster num: ", cluster_num)
    print(cluster_2_arm_dict)

    return cluster_2_arm_dict, arm_2_cluster_dict, arm_c_labels, affinity_m


def get_arm_clustering_spectral_clustering_with_affinity_m(A, affinity_m, cluster_num=20):
    sc = SpectralClustering(cluster_num, affinity='precomputed', assign_labels='kmeans', n_init=1000)

    sc.fit(affinity_m)
    arm_c_labels = sc.labels_

    #
    cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
    arm_2_cluster_dict = {j: None for j in range(A)}

    for a_i, c_label in enumerate(arm_c_labels):
        cluster_2_arm_dict[c_label].append(a_i)
        arm_2_cluster_dict[a_i] = c_label

    c_labels_hist = np.histogram(arm_c_labels, cluster_num)

    #
    print(c_labels_hist)
    print("Cluster num: ", cluster_num)
    print(cluster_2_arm_dict)

    return cluster_2_arm_dict, arm_2_cluster_dict, arm_c_labels


def get_arm_clustering_spectral_clustering_k_medoids(A, arm_graph, method=None, cluster_num=20, n_runs=1000,
                                                     n_components=-1, arm_features=None,
                                                     init_center_features=None):
    path_pairs_dict = dict(nx.all_pairs_shortest_path_length(arm_graph))
    cluster_2_arm_dict, arm_2_cluster_dict, total_inertia_seed, labels, last_centers = None, None, None, None, None
    if method == 'k_medoids':
        # ==================================================================================
        adj_matrix = np.array(nx.adjacency_matrix(arm_graph).todense())
        n_components = n_components
        n_components = cluster_num if n_components > 0 else n_components

        cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
        arm_2_cluster_dict = {j: None for j in range(A)}

        maps = spectral_embedding(
            adj_matrix,
            n_components=n_components,
            eigen_solver=None,
            random_state=None,
            eigen_tol=0.0,
            drop_first=False,
        )

        min_inertia = np.inf
        centers, labels = None, None
        for i in range(n_runs):
            kmedoids = KMedoids(n_clusters=cluster_num)
            kmedoids.fit(maps)
            this_inertia, this_labels, this_centers = kmedoids.inertia_, kmedoids.labels_, kmedoids.medoid_indices_

            if min_inertia > this_inertia:
                min_inertia = this_inertia
                centers = this_centers
                labels = this_labels

        cluster_hist = np.histogram(labels, cluster_num)

        #
        for a_i in range(A):
            cluster_2_arm_dict[labels[a_i]].append(a_i)
            arm_2_cluster_dict[a_i] = labels[a_i]

        #
        total_inertia_seed = 0.0
    # ------------------------------------------------------------------
    elif method == 'k_means':
        adj_matrix = np.array(nx.adjacency_matrix(arm_graph).todense())
        n_components = n_components
        n_components = cluster_num if n_components > 0 else n_components

        cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
        arm_2_cluster_dict = {j: None for j in range(A)}

        maps = spectral_embedding(
            adj_matrix,
            n_components=n_components,
            eigen_solver=None,
            random_state=None,
            eigen_tol=0.0,
            drop_first=False,
        )

        #
        init_centers = 'k-means++' if init_center_features is None else init_center_features
        last_centers, labels, inertia = k_means(
            maps, cluster_num,
            init=init_centers,
            n_init=n_runs, verbose=False
        )

        #
        for a_i in range(A):
            cluster_2_arm_dict[labels[a_i]].append(a_i)
            arm_2_cluster_dict[a_i] = labels[a_i]

        #
        total_inertia_seed = 0.0

    elif method == 'feature_k_means':
        cluster_2_arm_dict = {j: [] for j in range(cluster_num)}
        arm_2_cluster_dict = {j: None for j in range(A)}

        last_centers, labels, inertia = k_means(
            arm_features, cluster_num, n_init=n_runs, verbose=False
        )

        cluster_hist = np.histogram(labels, cluster_num)

        #
        for a_i in range(A):
            cluster_2_arm_dict[labels[a_i]].append(a_i)
            arm_2_cluster_dict[a_i] = labels[a_i]

        # Get concrete centers
        centers = []
        for c_i in range(cluster_num):
            this_center = last_centers[c_i].reshape(1, -1)
            this_arms = cluster_2_arm_dict[c_i]
            this_arm_feaures = arm_features[this_arms, :].reshape(len(this_arms), -1)
            distance_m = Kernel.rbf_kernel(this_center, this_arm_feaures, 5).reshape(-1, )
            index = np.argmax(distance_m)
            centers.append(this_arms[index])

        #
        total_inertia_seed = 0.0

    # print("Runs: ", n_runs, ", Min seed inertia: ", total_inertia_seed)

    return cluster_2_arm_dict, arm_2_cluster_dict, total_inertia_seed, labels, last_centers

