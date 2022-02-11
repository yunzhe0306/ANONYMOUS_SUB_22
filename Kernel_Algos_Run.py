from __future__ import division
import numpy as np
import pickle
import DataCreate.DataCreate as DC
import CrossValidation.CrossValidation as CV
import Algorithms.UCBTimeT as UCBT
import os
import warnings
import bandwidthselection.bandwidths as bwest

import sys
# warnings.filterwarnings("ignore")
import time
from datetime import datetime
import copy

start_time = time.time()


# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open("./New_logs/logfile_" + dt_string + ".log", "w")
        self.out = stdout
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        pass


sys.stdout = Logger(sys.stdout)

# Benchmark:38.4 for synthetic + 1/4


########################################################################################################################
# This is a main file.
#
#
#
#
########################################################################################################################
### List of algorithms
########################################################################################################################
# and GaussianKernels files in the respective libraries.
# algorithm_list = ['KTL-UCB-TaskSimEst', 'Lin-UCB-Ind']
# algorithm_list = ['KTL-UCB-TaskSimEst', 'Lin-UCB-Ind']
# algorithm_list = ['Lin-UCB-Ind']
# algorithm_list = ['KTL-UCB-TaskSimEst']
algorithm_list = ['Lin-UCB-Pool', 'KTL-UCB-TaskSimEst', 'Lin-UCB-Ind']

########################################################################################################################

# Hyper parameters
########################################################################################################################
# A = 100  # No of arms for synthetic data (data_flag = 3) Redundant for other flag

N_valid = 10  # No. of data points per arm in validation set
N = 1  # Algorithm starts with one random example assigned to each arm. This is a cold start problem.

# alpha = 1 for default settings, alpha = 0.2 for s_a,t
alpha = 1  # Hyper parameter for p_(a, t) ---- namely the beta

# T will be changed to different values for multi-class data
T = 4000

# 3 is syntehtic TL dataset, 7 real multiclass datasets (please see data_flag_multiclass)
# ------- 10 = Yelp data, 12 MovieLens, 77 Aug MNIST, 5 XRMB
data_flag = 12

crossval_flag = 0  # 0 if you don't need cross validation, 2 for batch learning way of doing cross valid,
# 3 for silverman's rule

Runs = 1  # Number of times we repeat the experiments.

# Flag for clustering
ACTIVATE_CLUSTERING_CLUB = True
# ALPHA_2 = 1

########################################################################################################################
INIT_N = N
INIT_T = T

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

Main_Program_flag = 0  # This variable is 0 when doing cross validation and 1 when running main block. Don't change this
# In future release we will get rid of this.

# Random seeds so that we use same sequence of data points while evaluating algorithms (Synthetic data ?)
randomSeedsTrain = np.array([15485867, 15486277, 15486727, 15487039,
                             15485917, 15486281, 15486739, 15487049,
                             15485927, 15486283, 15486749, 15487061,
                             15485933, 15486287, 15486769, 15487067,
                             15485941, 15486347, 15486773, 15487097,
                             15485959, 15486421, 15486781, 15487103,
                             15485989, 15486433, 15486791, 15487139,
                             15485993, 15486437, 15486803, 15487151,
                             15486013, 15486451, 15486827, 15487177,
                             15486041, 15486469, 15486833, 15487237,
                             15486047, 15486481, 15486857, 15487243,
                             15486059, 15486487, 15486869, 15487249,
                             15486071, 15486491, 15486871, 15487253])

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

########################################################################################################################

# cross validation --------------------------------

Parameter_Dict = dict()
for algorithm_flag in algorithm_list:
    if data_flag == 5:
        Parameter_Dict[algorithm_flag] = np.array([0.005, 5, 100, 0.001, 1])  # XRMB
    elif data_flag == 77:
        Parameter_Dict[algorithm_flag] = np.array([0.1, 2.15, 100, 0.001, 1])  # MNIST-Aug
    elif data_flag == 10:
        Parameter_Dict[algorithm_flag] = np.array([0.1, 50, 1000, 0.001, 1])  # Yelp
    elif data_flag == 12:
        Parameter_Dict[algorithm_flag] = np.array([0.1, 0.1, 200, 0.001, 1])  # MovieLens
    else:
        exit(1)

print(Parameter_Dict)

# pickle.dump(Parameter_Dict, open(str(data_flag) + 'parameter_results.p', "wb"))
CV_time = time.time() - start_time
print("Cross Validation took %s seconds ---" % (CV_time))

results_Folder = 'data_flag_' + str(data_flag)

# Save hyperparameters
Parameter_Dict_File_Name = 'cv_' + str(crossval_flag) + '_parameter_results.p'
Parameter_Dict_File_Path = os.path.join("ExperimentResults", results_Folder)
Parameter_Dict_File_Path = os.path.join(Parameter_Dict_File_Path, data_flag_multiclass)
if not os.path.exists(Parameter_Dict_File_Path):
    os.makedirs(Parameter_Dict_File_Path)
Parameter_Dict_File_Path = os.path.join(Parameter_Dict_File_Path, Parameter_Dict_File_Name)

pickle.dump(Parameter_Dict, open(Parameter_Dict_File_Path, "wb+"))

########################################################################################################################


### Main Algorithm Block
########################################################################################################################

start_time = time.time()

N = INIT_N
T = INIT_T

Main_Program_flag = 1  # This variable is 0 when doing cross validation and 1 when running main block. Don't change this.
# In future release we will get rid of this.
# Results_dict = dict()

# Initialization
AverageRegretRuns = np.zeros([len(algorithm_list), Runs])
AverageAccuracyRuns = np.zeros([len(algorithm_list), Runs])

"""
Run the algorithm
"""
for RunNumber in range(0, Runs):
    algo = 0
    regretUCBRuns = np.zeros([len(algorithm_list), T])
    # Test every algorithm
    # random seed
    rngTest = np.random.RandomState(randomSeedsTest[RunNumber])

    # Get the train data. This is just one example assigned to each arm randomly when N = 1 (cold start)
    Basic_DataXY = DC.TrainDataCollect(data_flag, A, N_valid, N, T, 1, RunNumber,
                                       Main_Program_flag)

    Basic_DataXY['N'] = N
    for algorithm_flag in algorithm_list:
        # Deep copy of DataXY for this algorithm
        DataXY = copy.deepcopy(Basic_DataXY)

        # Get the parameters
        bw_x, bw_prob, bw_prod, gamma, alpha = Parameter_Dict[algorithm_flag]
        print("Algorithm " + "Run number" + str(RunNumber))
        print(algorithm_flag, RunNumber, bw_x, bw_prob, bw_prod, gamma, alpha)

        # Run the bandit algorithm and get regret/reward with selected arm
        AverageRegret, AverageAccuracy, regretUCB, Selected_Arm_T, Exact_Arm_T, Task_sim_dict \
            = UCBT.ContextBanditUCBRunForTSteps(DataXY, T, data_flag,
                                                bw_x, bw_prob, bw_prod, gamma, alpha,
                                                algorithm_flag, RunNumber,
                                                activate_clustering=ACTIVATE_CLUSTERING_CLUB)

        # Store the result
        Results_dict = {'AverageRegret': AverageRegret, 'AverageAccuracy': AverageAccuracy, 'regretUCB': regretUCB,
                        'Selected_Arm_T': Selected_Arm_T, 'Exact_Arm_T': Exact_Arm_T, 'Task_sim_dict': Task_sim_dict}
        all_Results_File_Name = 'dataset_' + data_flag_multiclass + '_Run_' + str(RunNumber) + \
                                '_algorithm_' + algorithm_flag + '_Results_Dict.p'

        all_Results_File_Path = os.path.join("ExperimentResults", results_Folder)
        all_Results_File_Path = os.path.join(all_Results_File_Path, data_flag_multiclass)
        all_Results_File_Path = os.path.join(all_Results_File_Path, all_Results_File_Name)

        # Save Results for this algorithm in this run -----------------------
        pickle.dump(Results_dict, open(all_Results_File_Path, "wb"))
        AverageRegretRuns[algo, RunNumber] = AverageRegret
        AverageAccuracyRuns[algo, RunNumber] = AverageAccuracy
        regretUCBRuns[algo, :] = regretUCB

        # Print the current results -----------------------------------
        print("-" * 30, "\n")
        print("Current run number: ", RunNumber, "/", Runs)
        print("Current algorithm: ", algorithm_flag)
        print("Regret UCB: ", np.cumsum(regretUCB).tolist())
        print("-" * 10, "\n")
        print("Average regret: ", AverageRegret)
        print("Average accuracy: ", AverageAccuracy)
        print("Run time AverageAccuracyRuns: ", AverageAccuracyRuns)
        print("T = " + str(T))
        print("N = " + str(N))
        print("-" * 30, "\n")
        # Move to next algorithm
        algo += 1

    # Save the results into CSV files-----------------------------------
    save_Folder = os.path.join("ExperimentResults", results_Folder)
    save_Folder = os.path.join(save_Folder, data_flag_multiclass)
    save_Folder = os.path.join(save_Folder, "Excel")
    if not os.path.exists(save_Folder):
        os.makedirs(save_Folder)

    # Write the results to disk ------------------- regretUCB

    saveLocation = os.path.join(save_Folder, str(RunNumber) + ".csv")
    print("Writing to csv to directory: ", saveLocation)
    resultsFile = open(saveLocation, 'w+')
    for algoCounter in range(0, len(algorithm_list)):
        regrets = regretUCBRuns[algoCounter, :]
        regrets = np.cumsum(regrets).tolist()
        regrets_str = ",".join(map(str, regrets))
        resultsFile.write(regrets_str)
        resultsFile.write("\n")
    resultsFile.close()
    # -----------------------------------

CV_Time_File_Name = 'cv_' + str(crossval_flag) + '_cross_val_time_required.p'
CV_Time_File_Name_Path = os.path.join("ExperimentResults", results_Folder)
CV_Time_File_Name_Path = os.path.join(CV_Time_File_Name_Path, data_flag_multiclass)
CV_Time_File_Name_Path = os.path.join(CV_Time_File_Name_Path, CV_Time_File_Name)
pickle.dump(CV_time, open(CV_Time_File_Name_Path, "wb"))

print("-" * 30)
print("Final AverageAccuracyRuns: ", AverageAccuracyRuns)

Main_Block_Algo_time = time.time() - start_time

Training_Time_File_Name = 'dataset_' + data_flag_multiclass + '_Main_Block_time_required.p'

Training_Time_File_Path = os.path.join("ExperimentResults", results_Folder)
Training_Time_File_Path = os.path.join(Training_Time_File_Path, data_flag_multiclass)
Training_Time_File_Path = os.path.join(Training_Time_File_Path, Training_Time_File_Name)

pickle.dump(Main_Block_Algo_time, open(Training_Time_File_Path, "wb"))

print("Main Block of algorithm took %s seconds ---" % Main_Block_Algo_time)
