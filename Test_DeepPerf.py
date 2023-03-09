# 开发时间: 2023/2/20 10:57

import sys

sys.path.append('DeepPerf')
from RSFIN import *
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from DeepPerf.DeepPerf import DeepPerf as DP


def Stacking(YT, YD, YE, epoch=100, seed=0):
    YT = np.reshape(YT, (len(YT), 1))
    YD = np.reshape(YD, (len(YD), 1))
    YE = np.reshape(YE, (len(YE), 1))
    X = np.append(YT, YD, axis=1)
    Y = YE
    XY = np.append(X, Y, axis=1)

    anfis = ANFIS(XY)
    n = X.shape[1]

    lam = -0.00
    random.seed(seed)
    T_index = np.arange(len(Y))
    Train_index = random.sample(list(T_index), int(np.ceil(len(T_index) / 2)))
    Val_index = np.delete(T_index, Train_index)
    Rest_index = Val_index

    # Constructing rule layer based on training samples
    print('Constructing rules...\n')
    R_index = np.append(Train_index, Val_index)

    MR_b, Err = [], np.inf

    for i in trange(epoch):
        random.shuffle(R_index)
        anfis.MR = anfis.X[R_index[:2]]
        anfis.Train(Train_index, epoch=1, lam=lam)
        Err_t = anfis.Err_Rate(anfis.prediction(X[Val_index]), Y[Val_index], "All")

        if Err_t < Err:
            MR_b, Err = anfis.MR, Err_t
        if Err < 1:
            print('Complete rule construction in advance\n')
            break
    anfis.MR = MR_b

    # Training RSFIN
    print('\n Training model...')
    anfis.Train(np.append(Train_index, Val_index), 60, lam)
    Yp = anfis.prediction(X)

    return Yp

def run(SYSTEM, title):
    # SYSTEM = 'Apache'  # Apache, BDBC, BDBJ, x264, LLVM, SQL, hsmgp, hipacc, Dune, sac, javagc
    PATH = 'result/' + SYSTEM

    dfT = pd.read_csv(PATH + '_Train_' + title + '.csv')
    dfD = pd.read_csv(PATH + '_DiffTrain_' + title + '.csv')
    dfV = pd.read_csv(PATH + '_Test_' + title + '.csv')

    model = ANFIS([[1]])

    T_data = np.array(dfT)
    N_cross = int(np.ceil(T_data.shape[0] * 2 / 3))
    Train_data = T_data[0:N_cross, :]
    Vaild_data = T_data[N_cross:, :]
    Test_data = np.array(dfV)

    YE = Test_data[:, -1]
    YT, TimeT = DP(SYSTEM, Train_data, Vaild_data, Test_data, SYSTEM + '_' + title, seed=0)
    YT = np.reshape(YT, (len(YT),))

    Train_data = np.array(dfD)
    Vaild_data = np.array(dfT)

    YD, TimeD = DP(SYSTEM, Train_data, Vaild_data, Test_data, SYSTEM + '_' + title, seed=0)
    YD = np.reshape(YD, (len(YD),))

    YE = np.reshape(YE, (len(YE),))

    print(SYSTEM + ":" + " MRE = " + str(
        round(model.Err_Rate(YT, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YT, YE, "R2"), 3)))

    print(SYSTEM + ":" + " MRE = " + str(
        round(model.Err_Rate(YD, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YD, YE, "R2"), 3)))

    print("Ensemble:")
    lamb_e = 0
    mre_e, r2_e = np.inf, np.inf
    for i in range(1, 10):
        lamb = i / 10
        Y = YT * lamb + YD * (1 - lamb)
        mre = round(model.Err_Rate(Y, YE, "MRE"), 3)
        r2 = round(model.Err_Rate(Y, YE, "R2"), 3)
        print(SYSTEM + ": lamb = " + str(lamb) + ", MRE = " + str(
            mre) + ", R^2 = " + str(r2))
        if mre < mre_e:
            lamb_e, mre_e, r2_e = lamb, mre, r2

    YS = Stacking(YT, YD, YE)
    print("Stacking:")
    print(SYSTEM + ", MRE = " + str(
        round(model.Err_Rate(YS, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YS, YE, "R2"), 3)))

    result = dict()
    result["time_search_train"] = TimeT + TimeD
    result["base_MRE"] = round(model.Err_Rate(YT, YE, "MRE"), 3)
    result["base_R2"] = round(model.Err_Rate(YT, YE, "R2"), 3)
    result["diff_MRE"] = round(model.Err_Rate(YD, YE, "MRE"), 3)
    result["diff_R2"] = round(model.Err_Rate(YD, YE, "R2"), 3)
    result["lambda"] = lamb_e
    result["ensem_MRE"] = mre_e
    result["ensem_R2"] = r2_e
    result["stack_MRE"] = round(model.Err_Rate(YS, YE, "MRE"), 3)
    result["stack_R2"] = round(model.Err_Rate(YS, YE, "R2"), 3)
    result["Increase_ratio"] = round(1 - (min([round(model.Err_Rate(YD, YE, "MRE"), 3),
                                               mre_e, round(model.Err_Rate(YS, YE, "MRE"), 3)])
                                          / round(model.Err_Rate(YT, YE, "MRE"), 3)), 3)
    result = np.asarray(result).reshape(1)

    filename = 'DeepPerf/result/result_withEncoder_' + SYSTEM + '.csv'
    np.savetxt(filename, np.asarray(result), fmt="%s", delimiter=",")
    print('Save the statistics to file ' + filename + ' ...')

if __name__ == '__main__':
    nameList = ['Apache', 'BDBC', 'BDBJ', 'x264', 'LLVM', 'SQL', 'hsmgp', 'hipacc', 'Dune', 'sac', 'javagc']
    for SYSTEM in nameList:
        if SYSTEM in ['Apache',   'BDBC', 'BDBJ', 'x264', 'LLVM', 'SQL', 'hsmgp', 'hipacc', 'Dune']:
            continue
        run(SYSTEM, '5Nto50N')