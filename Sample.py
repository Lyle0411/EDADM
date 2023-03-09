#开发时间: 2023/2/15 22:31
from Diffusion_Design import Diffusion_Design as DD
import pandas as pd
import numpy as np
import random

def Packer(A, nameList, fileName):
    n_data = {}
    for i in range(len(nameList)):
        n_data[nameList[i]] = A[:, i]
    n_data = pd.DataFrame(n_data)
    n_data.to_csv(fileName, mode='w', index=0)


PATH = ''
SYSTEM = 'javagc' # Apache, BDBC, BDBJ, x264, LLVM, SQL, hsmgp, hipacc, Dune, sac, javagc
k = 5 # N
p = 50 # N, Number of diffusion samples

random.seed(0)
df = pd.read_csv(PATH + 'Data/' + SYSTEM + '_AllNumeric.csv')
name = list(df)
performance = np.array(df)
# if SYSTEM == 'sac':
#     performance = performance[random.sample(list(range(performance.shape[0])), 5000)] # just for sac
# if SYSTEM == 'javagc':
#     performance = performance[random.sample(list(range(performance.shape[0])), 5000)] # just for sac

N = performance.shape[1] - 1
pos = random.sample(list(range(performance.shape[0])), k*N)
remain = np.delete(np.array(list(range(performance.shape[0]))), pos, axis=0)

# Sample
sample = performance[pos]
remain_sample = performance[remain]
Packer(sample, name, PATH + 'result/'+ SYSTEM + "_Train_"+str(k)+"Nto"+str(p)+"N.csv")
Packer(remain_sample, name, PATH + 'result/'+ SYSTEM + "_Test_"+str(k)+"Nto"+str(p)+"N.csv")

int_mask = np.ones_like(performance[0])
int_mask[-1] = 0

# Diffusion
dd = DD(sample,N,p,num_epoch=int(10000))
x_0_new=dd.x_0_new

for i in range(x_0_new.shape[1]):
    if int_mask[i]:
        x_0_new[:, i] = np.round(x_0_new[:, i])

Packer(x_0_new, name, PATH + 'result/'+ SYSTEM + "_DiffTrain_"+str(k)+"Nto"+str(p)+"N.csv")


