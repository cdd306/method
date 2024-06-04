import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

def NLF(R, r, A, BC, M, N, d, f, lam, gam, flag):

    B = np.random.rand(d,f)
    C = np.random.rand(f,N)
    bc = B.dot(C)
    Error_NMAE = []
    Error_RMSE = []
    t = 1
    T = 1000  # 迭代次数

    while t < T:
        # 定义辅助矩阵
        AU = np.zeros((M, d), float)
        AD = np.zeros((M, d), float)
        BU = np.zeros((d, f), float)
        BD = np.zeros((d, f), float)
        CU = np.zeros((f, N), float)
        CD = np.zeros((f, N), float)

        I = 0
        err_nmae = 0
        err_rmse = 0

        TA = np.zeros((M, d), float)
        TB = np.zeros((d, f), float)
        TC = np.zeros((f, N), float)

        if t == 1:
            SA = A
            SB = B
            SC = C
            # 更新A
            for i in range(M):
                for k in range(d):
                    for j in range(N):
                        if R[i, j] != 0:
                            AU[i, k] = AU[i, k] + BC[k, j] * R[i, j]
                            AD[i, k] = AD[i, k] + BC[k, j] * r[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    A[i, k] = A[i, k] * AU[i, k] / (AD[i, k] + I * lam * A[i, k])
                    I = 0
            I = 0
            # 更新B
            for i in range(d):
                for k in range(f):
                    for j in range(N):
                        if R[i, j] != 0:
                            BU[i, k] = BU[i, k] + C[k, j] * BC[i, j]
                            BD[i, k] = BD[i, k] + C[k, j] * bc[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    B[i, k] = B[i, k] * BU[i, k] / (BD[i, k] + I * lam * B[i, k])
                    I = 0
            I = 0
            for k in range(f):
                for j in range(N):
                    for i in range(d):
                        if R[i, j] != 0:
                            CU[k, j] = CU[k, j] + B[i, k] * BC[i, j]
                            CD[k, j] = CD[k, j] + B[i, k] * bc[i, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    C[k, j] = C[k, j] * CU[k, j] / (CD[k, j] + I * lam * C[k, j])
                    I = 0
        else:
            # 更新A
            for i in range(M):
                for k in range(d):
                    for j in range(N):
                        if R[i, j] != 0:
                            AU[i, k] = AU[i, k] + BC[k, j] * R[i, j]
                            AD[i, k] = AD[i, k] + BC[k, j] * r[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TA[i, k] = gam * max(A[i, k] - SA[i, k], SA[i, k] - A[i, k])
                    SA[i, k] = A[i, k]
                    A[i, k] = A[i, k] * AU[i, k] / (AD[i, k] + I * lam * A[i, k]) + TA[i, k]
                    I = 0
            I = 0
            # 更新B
            for i in range(d):
                for k in range(f):
                    for j in range(N):
                        if R[i, j] != 0:
                            BU[i, k] = BU[i, k] + C[k, j] * BC[i, j]
                            BD[i, k] = BD[i, k] + C[k, j] * bc[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TB[i, k] = gam * max(B[i, k] - SB[i, k], SB[i, k] - B[i, k])
                    SB[i, k] = B[i, k]
                    B[i, k] = B[i, k] * BU[i, k] / (BD[i, k] + I * lam * B[i, k])
                    I = 0
            I = 0
            for k in range(f):
                for j in range(N):
                    for i in range(d):
                        if R[i, j] != 0:
                            CU[k, j] = CU[k, j] + B[i, k] * BC[i, j]
                            CD[k, j] = CD[k, j] + B[i, k] * bc[i, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TC[k, j] = gam * max(C[k, j] - SC[k, j], SC[k, j] - C[k, j])
                    TC[k, j] = C[k, j]
                    C[k, j] = C[k, j] * CU[k, j] / (CD[k, j] + I * lam * C[k, j])
                    I = 0

        bc = B.dot(C)
        r = A.dot(bc)
        print(r)

        I = 0
        # 计算评价指标——NMAE、RMSE
        for i in range(M):
            for j in range(N):
                if R[i, j] != 0:
                    I = I + 1
                    err_nmae = err_nmae + abs(R[i, j] - r[i, j])
                    err_rmse = err_rmse + (R[i, j] - r[i, j]) ** 2

        Error_NMAE.append(err_nmae / I)
        Error_RMSE.append(math.sqrt(err_rmse / I))

        t = t + 1

    return A, B, C, r, Error_NMAE, Error_RMSE,flag

five_scale_array = pd.read_csv('missing_of_five.csv')
R = np.array(five_scale_array)
print(R)

M = len(R)
N = len(R[0])
d = 10
f = 10
lam = 0.005
gam = 0.05

flag = np.zeros((M,N),float)

A = np.random.rand(M, d)
BC = np.random.rand(d, N)
r = A.dot(BC)

A,B,C,r,NMAE,RMSE,flag = NLF(R,r,A,BC,M,N,d,f,lam,gam,flag)
x = [i for i in range(999)]
print(RMSE)
plt.plot(x,RMSE)

plt.title('Method_Two')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.grid()

plt.show()

metric = pd.DataFrame([NMAE, RMSE], index=['NMAE', 'RMSE'])
metric = metric.T

print(metric)

metric.to_csv('metric/M2.csv', index=False)
