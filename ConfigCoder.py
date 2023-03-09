#开发时间: 2023/3/6 10:09
from Diffusion_Design import Diffusion_Design as DD
import numpy as np
import yaml
import pandas as pd
import random
import sys

sys.path.append('DeepPerf')
from RSFIN import *
from tqdm import tqdm, trange
from DeepPerf.DeepPerf import DeepPerf as DP

PATH = ''
class Coder:

    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = Y
        self.mu = np.mean(self.X, axis=1)
        self.cov = np.cov(self.X)
        self.covp = np.linalg.pinv(self.cov)
        self.AnchorChoice()
        self.Encode()

    def Md(self, x1, x2):
        term1 = np.reshape(x1-x2, (len(x1),1))
        term2 = self.covp

        return float(np.sqrt(np.dot(np.dot(term1.transpose(),term2), term1)))

    def g_fun(self, d):
        return np.exp(-((d-self.d_mu)/self.d_sigma)**2)

    def Encode(self):
        dis = []
        for i in range(self.X.shape[1]):
            dis.append([self.Md(self.X[:,i], ap) for ap in self.Ap])
        self.d_mu = np.mean(dis)
        self.d_sigma = np.std(dis)

        code = []
        for i in range(self.X.shape[1]):
            code_temp = self.Encoder(self.X[:,i])
            code.append(self.Encoder(self.X[:,i]))

        self.code = np.array(code)

    def Encoder(self, x):

        d = []
        for ap in self.Ap:
            d.append(self.Md(x, ap))

        d = np.array([self.g_fun(d_i) for d_i in d])

        return d

    def MeanShift(self):
        y_m = np.mean(self.Y)
        Y = self.Y - y_m

        y_mean = np.linspace(min(Y), max(Y), 20)
        r = (max(Y)-min(Y))/8

        y_mean_t = np.zeros_like(y_mean)
        while np.mean(abs(y_mean-y_mean_t)) > 0.01:
            y_mean_t = np.copy(y_mean)
            for i in range(len(y_mean)):
                y = y_mean[i]
                pos = np.where(np.abs(Y-y)<r)
                if len(pos[0]) == 0:
                    y_mean[i] = -1
                    continue
                else:
                    y_mean[i] = round(np.mean(Y[pos]),3)
        y_mean = np.delete(y_mean, np.where(y_mean==-1),0)

        self.y_c = np.unique(y_mean) + y_m

    def AnchorChoice(self):
        # ap_pos = random.sample(list(range(len(Y))), 2)
        # self.Ap = self.X[:, ap_pos].transpose()
        self.MeanShift()
        cl = np.zeros_like(self.y_c)
        for y in self.Y:
            dis = np.abs(y-self.y_c)
            pos = np.where(dis == min(dis))
            cl[pos] += 1
        c_n = np.max([np.min(cl),5])
        Ap = []
        for y in self.y_c:
            dis = abs(self.Y - y)
            sdis = np.sort(dis)
            pos = np.where(dis<=sdis[int(c_n)])
            ap = np.mean([self.X[:,pos[0][i]] for i in range(len(pos[0]))], axis=0)
            Ap.append(ap)


        self.Ap = Ap

    def __str__(self):
        msg = {}
        Md_msg = {}

        msg['Data_size'] = str(self.X.shape).replace('\n','')
        Md_msg['mu'] = [float(item) for item in self.mu]
        Md_msg['cov'] = [str(X) for X in self.cov]

        msg['Maha distance'] = Md_msg

        return yaml.dump(msg)


    def Packer(self, fileName):
        A = np.append(self.code, np.reshape(self.Y, (len(self.Y),1)), axis=1)
        nameList = []
        for i in range(len(self.y_c)):
            nameList.append('feature_' + str(i))
        nameList.append('PERF')

        n_data = {}
        for i in range(len(nameList)):
            n_data[nameList[i]] = A[:, i]
        n_data = pd.DataFrame(n_data)
        n_data.to_csv(fileName, mode='w', index=0)

def Packer(A, nameList, fileName):
    n_data = {}
    for i in range(len(nameList)):
        n_data[nameList[i]] = A[:, i]
    n_data = pd.DataFrame(n_data)
    n_data.to_csv(fileName, mode='w', index=0)

def Sample(patho, pathr, paths, k, p):
    dfo = pd.read_csv(patho)
    nameo = list(dfo)
    XYo = np.array(dfo)
    N = XYo.shape[1] - 1

    df = pd.read_csv(pathr)
    name = list(df)
    XY = np.array(df)
    # N = XY.shape[1] - 1
    pos = random.sample(list(range(XY.shape[0])), k*N)
    remain = np.delete(np.array(list(range(XY.shape[0]))), pos, axis=0)

    # Sample
    sampleo = XYo[pos]
    remain_sampleo = XYo[remain]
    sample = XY[pos]
    remain_sample = XY[remain]
    Packer(sampleo, nameo, paths + '_Train.csv')
    Packer(remain_sampleo, nameo, paths + '_Test.csv')
    Packer(sample, name, paths + '_code_Train.csv')
    Packer(remain_sample, name, paths + '_code_Test.csv')
    dd = DD(sample, N, p, num_epoch=int(10000))
    x_0_new = dd.x_0_new
    for x_0 in x_0_new:
        x = x_0[:-1]
        x_0[np.where(x < 0)] = 0.00001
        x_0[np.where(x > 1)] = 1
    Packer(x_0_new, name, paths + "_code_DiffTrain_" + str(k) + "Nto" + str(p) + "N.csv")

def Encode():
    nameList = ['Apache', 'BDBC', 'BDBJ', 'x264', 'LLVM', 'SQL', 'hsmgp', 'hipacc', 'Dune', 'sac', 'javagc']
    for SYSTEM in nameList:
        df = pd.read_csv(PATH + 'Data/' + SYSTEM + '_AllNumeric.csv')
        data = np.array(df)
        X = data[:, :-1].transpose()
        Y = data[:, -1]
        coder = Coder(X, Y)
        coder.Packer(PATH + 'encode_result/' + SYSTEM + "_code.csv")

def Sample_encode(k, p):
    nameList = ['Apache', 'BDBC', 'BDBJ', 'x264', 'LLVM', 'SQL', 'hsmgp', 'hipacc', 'Dune', 'sac', 'javagc']
    for SYSTEM in nameList:
        random.seed(0)

        pathr = PATH + 'encode_result/' + SYSTEM + "_code.csv"
        paths = PATH + 'encode_result/' + SYSTEM
        patho = PATH + 'Data/' + SYSTEM + '_AllNumeric.csv'
        Sample(patho, pathr, paths, k, p)

def Stacking(YT, YD, YE, size, seed=0):
    YT = np.reshape(YT, (len(YT), 1))
    YD = np.reshape(YD, (len(YD), 1))
    YE = np.reshape(YE, (len(YE), 1))
    X = np.append(YT, YD, axis=1)
    Y = YE
    data = np.append(X, Y, axis=1)

    random.seed(seed)
    T_index = np.arange(len(Y))

    Train_index = random.sample(list(T_index), size)

    T_data = data[Train_index]
    N_cross = int(np.ceil(T_data.shape[0] * 2 / 3))
    Train_data = T_data[0:N_cross, :]
    Vaild_data = T_data[N_cross:, :]
    Test_data = np.copy(data)

    Yp, TimeT = DP(SYSTEM, Train_data, Vaild_data, Test_data, '_', seed=0)
    Yp = np.reshape(Yp, (len(Yp),))

    return Yp
def MRE(Y_test, Y_pred_test):
    return np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel())))*100
def run(SYSTEM, title):
    # SYSTEM = 'Apache'  # Apache, BDBC, BDBJ, x264, LLVM, SQL, hsmgp, hipacc, Dune, sac, javagc
    PATH = 'encode_result/' + SYSTEM

    dofT = pd.read_csv(PATH + '_Train.csv')
    dofV = pd.read_csv(PATH + '_Test.csv')

    T_data = np.array(dofT)
    N_cross = int(np.ceil(T_data.shape[0] * 2 / 3))
    Train_datao = T_data[0:N_cross, :]
    Vaild_datao = T_data[N_cross:, :]
    Test_data = np.array(dofV)

    YT, TimeT = DP(SYSTEM, Train_datao, Vaild_datao, np.append(T_data, Test_data, axis=0), SYSTEM + '_' + title, seed=0)
    YTa = np.reshape(YT, (len(YT),))
    YoT = YTa[len(T_data):]

    dfT = pd.read_csv(PATH + '_code_Train.csv')
    dfD = pd.read_csv(PATH + '_code_DiffTrain_' + title + '.csv')
    dfV = pd.read_csv(PATH + '_code_Test.csv')

    model = ANFIS([[1]])

    T_data = np.array(dfT)
    Train_data = T_data[0:N_cross, :]
    Vaild_data = T_data[N_cross:, :]
    Test_data = np.array(dfV)

    YE = Test_data[:, -1]
    YT, nothing = DP(SYSTEM, Train_data, Vaild_data, Test_data, SYSTEM + '_' + title, seed=0)
    YT = np.reshape(YT, (len(YT),))

    Train_datad = np.array(dfD)
    Vaild_datad = np.array(dfT)

    All_data = np.append(np.append(Train_datad, Vaild_datad, axis=0), Test_data, axis=0)

    YTd, TimeD = DP(SYSTEM, Train_datad, Vaild_datad, All_data, SYSTEM + '_' + title, seed=0)
    YTd = np.reshape(YTd, (len(YTd),))
    YD = YTd[(len(Train_datad) + len(Vaild_datad)):]

    YE = np.reshape(YE, (len(YE),))

    print(SYSTEM + ":" + " MRE = " + str(
        round(model.Err_Rate(YoT, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YoT, YE, "R2"), 3)))

    print(SYSTEM + ":" + " MRE = " + str(
        round(model.Err_Rate(YD, YE, "MRE"), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YD, YE, "R2"), 3)))

    print("Ensemble:")
    lamb_e = 0
    mre_e, r2_e = np.inf, np.inf
    for i in range(1, 10):
        lamb = i / 10
        Y = YoT * lamb + YD * (1 - lamb)
        mre = round(MRE(YE, Y), 3)
        r2 = round(model.Err_Rate(Y, YE, "R2"), 3)
        print(SYSTEM + ": lamb = " + str(lamb) + ", MRE = " + str(
            mre) + ", R^2 = " + str(r2))
        if mre < mre_e:
            lamb_e, mre_e, r2_e = lamb, mre, r2

    YTa = np.reshape(YTa, (len(YTa), 1))
    YTd = np.reshape(YTd, (len(YTd), 1))


    Train_data = np.append(np.append(YTa[0:N_cross], YTd[len(Train_datad):len(Train_datad)+N_cross],
                                     axis=1), np.reshape(Train_datao[:,-1], (len(Train_datao[:,-1]), 1)), axis=1)
    Vaild_data = np.append(np.append(YTa[N_cross:len(T_data)],
                                     YTd[len(Train_datad)+N_cross:len(Train_datad)+len(T_data)], axis=1),
                           np.reshape(Vaild_datao[:,-1], (len(Vaild_datao[:,-1]), 1)), axis=1)
    Test_data = np.append(np.append(np.reshape(YoT, (len(YoT), 1)),
                                    np.reshape(YD, (len(YD), 1)), axis=1), np.reshape(YE, (len(YE), 1)), axis=1)
    YS, TimeS = DP(SYSTEM, Train_data, Vaild_data, Test_data, SYSTEM + '_' + title, seed=0)

    print("Stacking:")
    print(SYSTEM + ", MRE = " + str(
        round(MRE(YE, YS), 3)) + ", R^2 = " + str(
        round(model.Err_Rate(YS, YE, "R2"), 3)))

    result = dict()
    result["time_search_train"] = TimeT + TimeD + TimeS
    result["base_MRE"] = round(MRE(YE, YoT), 3)
    result["base_R2"] = round(model.Err_Rate(YoT, YE, "R2"), 3)
    result["code_MRE"] = round(MRE(YE, YT), 3)
    result["code_R2"] = round(model.Err_Rate(YT, YE, "R2"), 3)
    result["diff_MRE"] = round(MRE(YE, YD), 3)
    result["diff_R2"] = round(model.Err_Rate(YD, YE, "R2"), 3)
    result["lambda"] = lamb_e
    result["ensem_MRE"] = mre_e
    result["ensem_R2"] = r2_e
    result["stack_MRE"] = round(MRE(YE, YS), 3)
    result["stack_R2"] = round(model.Err_Rate(YS, YE, "R2"), 3)
    result["Increase_ratio"] = round(1 - (min([round(MRE(YE, YD), 3),
                                               mre_e, round(MRE(YE, YS), 3)])
                                          / round(model.Err_Rate(YoT, YE, "MRE"), 3)), 3)
    result = np.asarray(result).reshape(1)

    filename = 'encode_result/result/result_withEncoder_' + title + '_' + SYSTEM + '.csv'
    np.savetxt(filename, np.asarray(result), fmt="%s", delimiter=",")
    print('Save the statistics to file ' + filename + ' ...')

    YE = np.reshape(YE, (len(YE),1))
    YoT = np.reshape(YoT, (len(YE), 1))
    YT = np.reshape(YT, (len(YE), 1))
    YD = np.reshape(YD, (len(YE), 1))
    result = np.copy(YE)
    result = np.append(result, YoT, axis=1)
    result = np.append(result, YT, axis=1)
    result = np.append(result, YD, axis=1)

    name = ["ground truth", "Original", "Code", "Code+Diffusion"]


    Packer(result, name, 'encode_result/result/prediction_'+ title + '_' + SYSTEM + '.csv')


if __name__ == "__main__":
    # Sample_encode(5, 20)
    nameList = ['Apache', 'BDBC', 'BDBJ', 'x264', 'LLVM', 'SQL', 'hsmgp', 'hipacc', 'Dune', 'sac', 'javagc']
    for SYSTEM in nameList:
        run(SYSTEM, '5Nto20N')

