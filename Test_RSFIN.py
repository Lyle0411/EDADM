# 开发时间: 2023/2/16 16:59

from RSFIN import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def Test(T, V, epoch, seed=0):
    XY = np.append(np.array(T), np.array(V), axis=0)
    lenT, lenV = len(T), len(V)

    XY[:, -1] = XY[:, -1] + 0.1 * max(XY[:, -1])  # Reduce the zero sensitivity of MRE
    X = XY[:, 0:-1]
    Y = XY[:, -1]

    anfis = ANFIS(XY)
    n = X.shape[1]

    lam = -0.00
    random.seed(seed)
    Rest_index = np.arange(lenT, lenT + lenV)
    T_index = np.arange(lenT)
    # Train_index = random.sample(list(T_index), int(np.ceil(len(T_index) / 2)))
    # Val_index = np.delete(T_index, Train_index)
    Train_index = T_index
    Val_index = Rest_index

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
    Yp = anfis.prediction(X[Rest_index])

    print(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
         round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", R^2 = " + str(
        round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))

    return Yp,Y[Rest_index]
    # return anfis.Err_Rate(Yp, Y[Rest_index], "MRE")
    # Drawing comparison diagram
    # plt.plot(np.array(list(range(len(Yp)))) + 1, Y[Rest_index])
    # plt.plot(np.array(list(range(len(Yp)))) + 1, Yp)
    # plt.legend(["real", "prediction"])
    # plt.title(SYSTEM + ", N = " + str(len(Val_index) + len(Train_index)) + ", MRE = " + str(
    #     round(anfis.Err_Rate(Yp, Y[Rest_index], "MRE"), 3)) + ", $R^2$ = " + str(
    #     round(anfis.Err_Rate(Yp, Y[Rest_index], "R2"), 3)))
    # plt.xlabel('Configuration index')
    # plt.ylabel('Performance')
    # plt.show()

def Stacking(YT, YD, YE, epoch=100, seed=0):
    YT = np.reshape(YT, (len(YT),1))
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

if __name__ == '__main__':
    SYSTEM = 'Apache' # Apache, BDBC, BDBJ, x264, LLVM, SQL, hsmgp, hipacc, Dune, sac, javagc
    PATH = 'result/' + SYSTEM
    title = '5Nto50N'

    dfT = pd.read_csv(PATH + '_Train_' + title + '.csv')
    dfD = pd.read_csv(PATH + '_DiffTrain_' + title + '.csv')
    dfV = pd.read_csv(PATH + '_Test_' + title + '.csv')

    model = ANFIS([[1]])

    YT,YE = Test(dfT, dfV, epoch = 100)
    YD,YE = Test(dfD, dfV, epoch = 100)
    print(YT, YT.shape, type(YT))


    # lamb = 0.4
    print("Ensemble:")
    for i in range(1,10):
        lamb = i/10
        Y = YT*lamb + YD*(1-lamb)
        print(SYSTEM +  ": lamb = "+ str(lamb)+", MRE = " + str(
        round(model.Err_Rate(Y, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(Y, YE, "R2"), 3)))


    YS = Stacking(YT, YD, YE)
    print("Stacking:")
    print(SYSTEM + ", MRE = " + str(
        round(model.Err_Rate(YS, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YS, YE, "R2"), 3)))


