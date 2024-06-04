import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

def NLF(R, A, B, M, N, d, r, lam, gam, flag):
    Error_NMAE = []
    Error_RMSE = []
    t = 1
    T = 1000  # 迭代次数

    while t < T:
        # 定义辅助矩阵
        AU = np.zeros((M, d), float)
        AD = np.zeros((M, d), float)
        BU = np.zeros((d, N), float)
        BD = np.zeros((d, N), float)
        I = 0
        err_nmae = 0
        err_rmse = 0

        TA = np.zeros((M, d), float)
        TB = np.zeros((d, N), float)


        # 更新A
        if t == 1:
            SA = A
            SB = B
            # 更新A
            for i in range(M):
                for k in range(d):
                    for j in range(N):
                        if R[i, j] != 0:
                            AU[i, k] = AU[i, k] + B[k, j] * R[i, j]
                            AD[i, k] = AD[i, k] + B[k, j] * r[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    A[i, k] = A[i, k] * AU[i, k] / (AD[i, k] + I * lam * A[i, k])

            I = 0
            # 更新B
            for k in range(d):
                for j in range(N):
                    for i in range(M):
                        if R[i, j] != 0:
                            BU[k, j] = BU[k, j] + A[i, k] * R[i, j]
                            BD[k, j] = BD[k, j] + A[i, k] * r[i, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    B[k, j] = B[k, j] * BU[k, j] / (BD[k, j] + I * lam * B[k, j])

        else:
            # 更新A
            for i in range(M):
                for k in range(d):
                    for j in range(N):
                        if R[i, j] != 0:
                            AU[i, k] = AU[i, k] + B[k, j] * R[i, j]
                            AD[i, k] = AD[i, k] + B[k, j] * r[i, j]  # 上一次迭代预测的r
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TA[i, k] = gam * max(A[i, k] - SA[i, k], SA[i, k] - A[i, k])
                    SA[i, k] = A[i, k]
                    A[i, k] = A[i, k] * AU[i, k] / (AD[i, k] + I * lam * A[i, k]) + TA[i,k]

            I = 0
            # 更新B
            for k in range(d):
                for j in range(N):
                    for i in range(M):
                        if R[i, j] != 0:
                            BU[k, j] = BU[k, j] + A[i, k] * R[i, j]
                            BD[k, j] = BD[k, j] + A[i, k] * r[i, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TB[k, j] = gam * max(B[k, j] - SB[k, j], SB[k, j] - B[k, j])
                    SB[k, j] = B[k, j]
                    B[k, j] = B[k, j] * BU[k, j] / (BD[k, j] + I * lam * B[k, j]) + TB[k, j]


        r = A.dot(B)  # 预测值r

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

    return A, B, Error_NMAE, Error_RMSE, r, flag

five_scale_array = pd.read_csv('missing_of_five.csv')
R = np.array(five_scale_array)
print(R)

M = len(R)
N = len(R[0])
d = 2
lam = 0.06
gam = 0.06

flag = np.zeros((M,N),float)

# 初始化A,B
A = np.random.rand(M, d)
B = np.random.rand(d, N)

r = A.dot(B)

A,B,NMAE,RMSE,r,flag = NLF(R, A, B, M, N, d, r, lam, gam, flag)

x = [i for i in range(999)]
print(RMSE)
plt.plot(x,RMSE)

plt.title('Method_One')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.grid()

plt.show()

metric = pd.DataFrame([NMAE, RMSE], index=['NMAE', 'RMSE'])
metric = metric.T

print(metric)

metric.to_csv('metric/M1.csv', index=False)