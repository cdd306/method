import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3.TF——矩阵三分解
def NLF(R, r, A, B, C, M, N, d, f, learning_rate, lam, gam, flag):
    Error_NMAE = []
    Error_RMSE = []
    t = 1
    T = 1000  # 迭代次数

    while t < T:
        # 定义辅助矩阵
        AU = np.zeros((M, d), float)
        BC = np.zeros((M, d), float)

        BU = np.zeros((d, f), float)

        CU = np.zeros((f, N), float)
        AB = np.zeros((f, N), float)

        err_nmae = 0
        err_rmse = 0
        I = 0

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
                        if R[i][j] != 0:
                            for l in range(f):
                                BC[i, k] = BC[i, k] + sigmoid(B[k, l]) * sigmoid(C[l, j])
                            AU[i, k] = AU[i, k] + (R[i, j] - r[i, j]) * BC[i, k]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    A[i, k] = A[i, k] + learning_rate * sigmoid(A[i, k]) * (1 - sigmoid(A[i, k])) * (AU[i, k] - I * lam * sigmoid(A[i, k]))
                I = 0

            I = 0

            # 更新B
            for k in range(d):
                for l in range(f):
                    for i in range(M):
                        for j in range(N):
                            if R[i][j] != 0:
                                BU[k, l] = BU[k, l] + (R[i, j] - r[i, j]) * sigmoid(A[i, k]) * sigmoid(C[l, j])
                                I = I + 1
                            else:
                                flag[i][j] = 1
                    B[k, l] = B[k, l] + learning_rate * sigmoid(B[k, l]) * (1 - sigmoid(B[k, l])) * (BU[k, l] - lam * I * sigmoid(B[k, l]))
                    I = 0

            I = 0

            # 更新C
            for l in range(f):
                for j in range(N):
                    for i in range(M):
                        if R[i][j] != 0:
                            for k in range(d):
                                AB[l, j] = AB[l, j] + sigmoid(A[i, k]) * sigmoid(B[k, l])
                            CU[l, j] = CU[l, j] + (R[i, j] - r[i, j]) * AB[l, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    C[l, j] = C[l, j] + learning_rate * sigmoid(C[l, j]) * (1 - sigmoid(C[l, j])) * (CU[l, j] - lam * I * sigmoid(C[l, j]))
                    I = 0
        else:
            # 更新A
            for i in range(M):
                for k in range(d):
                    for j in range(N):
                        if R[i][j] != 0:
                            for l in range(f):
                                BC[i, k] = BC[i, k] + sigmoid(B[k, l]) * sigmoid(C[l, j])
                            AU[i, k] = AU[i, k] + (R[i, j] - r[i, j]) * BC[i, k]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TA[i, k] = gam * max(A[i,k] - SA[i, k], SA[i, k] - A[i, k])
                    SA[i, k] = A[i, k]
                    A[i, k] = A[i, k] + learning_rate * sigmoid(A[i, k]) * (1 - sigmoid(A[i, k])) * (AU[i, k] - I * lam * sigmoid(A[i, k])) + TA[i, k]
                I = 0

            I = 0

            # 更新B
            for k in range(d):
                for l in range(f):
                    for i in range(M):
                        for j in range(N):
                            if R[i][j] != 0:
                                BU[k, l] = BU[k, l] + (R[i, j] - r[i, j]) * sigmoid(A[i, k]) * sigmoid(C[l, j])
                                I = I + 1
                            else:
                                flag[i][j] = 1
                    TB[k, l] = gam * max(B[k, l] - SB[k, l], SB[k, l] - B[k, l])
                    SB[k, l] = B[k, l]
                    B[k, l] = B[k, l] + learning_rate * sigmoid(B[k, l]) * (1 - sigmoid(B[k, l])) * (BU[k, l] - lam * I * sigmoid(B[k, l])) + TB[k, l]
                    I = 0

            I = 0

            # 更新C
            for l in range(f):
                for j in range(N):
                    for i in range(M):
                        if R[i][j] != 0:
                            for k in range(d):
                                AB[l, j] = AB[l, j] + sigmoid(A[i, k]) * sigmoid(B[k, l])
                            CU[l, j] = CU[l, j] + (R[i, j] - r[i, j]) * AB[l, j]
                            I = I + 1
                        else:
                            flag[i][j] = 1
                    TC[l, j] = gam * max(C[l, j] - SC[l, j], SC[l, j] - C[l, j])
                    SC[l, j] = C[l, j]
                    C[l, j] = C[l, j] + learning_rate * sigmoid(C[l, j]) * (1 - sigmoid(C[l, j])) * (CU[l, j] - lam * I * sigmoid(C[l, j])) + TC[l, j]
                    I = 0

        r = sigmoid(A).dot(sigmoid(B)).dot(sigmoid(C))
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

    return Error_NMAE, Error_RMSE, flag

five_scale_array = pd.read_csv('missing_of_five.csv')
R = np.array(five_scale_array)
print(R)

M = len(R)
N = len(R[0])
d = 3
f = 3
learning_rate = 0.001
lam = 0.05
gam = 0.5

flag = np.zeros((M,N),float)

# 初始化A,B,C
A = np.random.rand(M, d)
B = np.random.rand(d, f)
C = np.random.rand(f, N)

r = sigmoid(A).dot(sigmoid(B)).dot(sigmoid(C))

NMAE,RMSE,flag = NLF(R, r, A, B, C, M, N, d, f, learning_rate, lam, gam, flag)

x = [i for i in range(999)]
print(RMSE)
plt.plot(x,RMSE)

plt.title('Method_Three')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.grid()

plt.show()

metric = pd.DataFrame([NMAE, RMSE], index=['NMAE', 'RMSE'])
metric = metric.T

print(metric)

metric.to_csv('metric/M3.csv', index=False)