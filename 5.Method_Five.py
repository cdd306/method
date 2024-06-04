import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_R = pd.read_csv('missing_of_five.csv')
df_R_ = pd.read_csv('missing_of_three.csv')
R = np.array(df_R)
R_ = np.array(df_R_)

I = len(R)
J = len(R[0])
e = 5

# 初始化A,B,C,H,H_,r,r_
A = np.random.rand(I, e)
B = np.random.rand(I, e)
C = np.random.rand(I, e)
H = np.random.rand(e, e)
H_ = np.random.rand(e, e)

P = np.random.rand(1, I)
Q = np.random.rand(1, I)
# print(P[0,1])
# print(Q)

o = (A + B).dot(H).dot((A + B).T)
o_ = (A + C).dot(H_).dot((A + C).T)

# 映射到三分制
def map_to_3_scale(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # 将数组映射到0到1的范围内
    mapped_arr = np.interp(scaled_arr, (0,1), (1,3))  # 将数组映射到1到3的范围内
    return mapped_arr
# 映射到五分制
def map_to_5_scale(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # 将数组映射到0到1的范围内
    mapped_arr = np.interp(scaled_arr, (0,1), (1,5))
    return mapped_arr

print(o)
print(o_)

def NLF(R, R_, o, o_, I, J, e, A, B, C, P, Q, H, H_):

    lam = 0.001
    beta = 0.05
    gam = 0.25
    t = 1
    T = 1000
    Error_NMAE = []
    Error_RMSE = []


    while t < T:
        p = np.zeros((J, e), float)
        p_ = np.zeros((J, e), float)
        AU = np.zeros((I, e), float)
        AD = np.zeros((I, e), float)
        BU = np.zeros((I, e), float)
        BD = np.zeros((I, e), float)
        CU = np.zeros((I, e), float)
        CD = np.zeros((I, e), float)
        HU = np.zeros((e, e), float)
        HD = np.zeros((e, e), float)
        H_U = np.zeros((e, e), float)
        H_D = np.zeros((e, e), float)

        PU = np.zeros((1, I), float)
        PD = np.zeros((1, I), float)
        QU = np.zeros((1, I), float)
        QD = np.zeros((1, I), float)

        TA = np.zeros((I, e), float)
        TB = np.zeros((I, e), float)
        TC = np.zeros((I, e), float)
        TH = np.zeros((e, e), float)
        TH_ = np.zeros((e, e), float)
        TP = np.zeros((1, I), float)
        TQ = np.zeros((1, I), float)

        err_nmae = 0
        err_rmse = 0

        if t == 1:
            SA = A
            SB = B
            SC = C
            SH = H
            SH_ = H_
            SP = P
            SQ = Q
            # 计算p,p_
            for i in range(I):
                for m in range(e):
                    for j in range(J):
                        for n in range(e):
                            if R[i, j] != 0:
                                p[j, m] = p[j, m] + H[m, n] * (A[j, n] + B[j, n])
                                p_[j, m] = p_[j, m] + H_[m, n] * (A[j, n] + C[j, n])

            # 更新A,B,C
            for i in range(I):
                for m in range(e):
                    for j in range(J):
                        if R[i,j] != 0:
                            AU[i, m] = AU[i, m] + R[i, j] * p[j, m] + beta * R_[i, j] * p_[j, m]
                            AD[i, m] = AD[i, m] + (P[0,i] + P[0,j] + o[i, j]) * p[j, m] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * p_[j, m] + lam * A[i, m]
                            BU[i, m] = BU[i, m] + R[i, j] * p[j, m]
                            BD[i, m] = BD[i, m] + (P[0,i] + P[0,j] + o[i, j]) * p[j, m] + lam * B[i, m]
                            CU[i, m] = CU[i, m] + beta * R_[i, j] * p_[j, m]
                            CD[i, m] = CD[i, m] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * p_[j, m] + lam * C[i, m]
                    A[i, m] = A[i, m] * AU[i, m] / AD[i, m]
                    B[i, m] = B[i,m] * BU[i, m] / BD[i, m]
                    C[i, m] = C[i, m] * CU[i, m] / CD[i, m]

            # 更新H,H_
            for m in range(e):
                for n in range(e):
                    for i in range(I):
                        for j in range(J):
                            if R[i, j] != 0:
                                seda = A[i, m] * A[j, n] + A[i, m] * B[j, n] + B[i, m] * A[j, n] + B[i, m] * B[j, n]
                                seda_ = A[i, m] * A[j, n] + A[i, m] * C[j, n] + C[i, m] * A[j, n] + C[i, m] * C[j, n]
                                HU[m, n] = HU[m, n] + R[i, j] * seda
                                HD[m, n] = HD[m,n] + (P[0,i] + P[0,j] + o[i, j]) * seda + lam * H[m, n]
                                H_U[m, n] = H_U[m,n] + beta * R_[i, j] * seda_
                                H_D[m, n] = H_D[m,n] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * seda_ + lam * H_[m, n]
                    H[m, n] = H[m, n] * HU[m, n] / HD[m, n]
                    H_[m, n] = H_[m, n] * H_U[m, n] / H_D[m, n]

            # 更新P,Q
            for i in range(I):
                for j in range(J):
                    if R[i,j] != 0:
                        PU[0, i] = PU[0, i] + R[i, j]
                        PD[0, i] = PD[0, i] + P[0, i] + P[0, j] + o[i, j] + lam * P[0, i]
                        QU[0, i] = QU[0, i] + beta * R_[i, j]
                        QD[0, i] = QD[0, i] + beta * (Q[0, i] + Q[0, j] + o_[i, j]) + lam * Q[0, i]
                P[0, i] = P[0, i] * PU[0, i] / PD[0, i]
                Q[0, i] = Q[0, i] * QU[0, i] / QD[0, i]

        ## 当t>=2
        else:
            # 计算p,p_
            for i in range(I):
                for m in range(e):
                    for j in range(J):
                        for n in range(e):
                            if R[i, j] != 0:
                                p[j, m] = p[j, m] + H[m, n] * (A[j, n] + B[j, n])
                                p_[j, m] = p_[j, m] + H_[m, n] * (A[j, n] + C[j, n])
            # 更新A,B,C
            for i in range(I):
                for m in range(e):
                    for j in range(J):
                        if R[i, j] != 0:
                            AU[i, m] = AU[i, m] + R[i, j] * p[j, m] + beta * R_[i, j] * p_[j, m]
                            AD[i, m] = AD[i, m] + (P[0,i] + P[0,j] + o[i, j]) * p[j, m] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * p_[j, m] + lam * A[i, m]
                            BU[i, m] = BU[i, m] + R[i, j] * p[j, m]
                            BD[i, m] = BD[i, m] + (P[0,i] + P[0,j] + o[i, j]) * p[j, m] + lam * B[i, m]
                            CU[i, m] = CU[i, m] + beta * R_[i, j] * p_[j, m]
                            CD[i, m] = CD[i, m] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * p_[j, m] + lam * C[i, m]
                    TA[i,m] = gam * max(A[i, m] - SA[i, m], SA[i, m] - A[i, m])
                    TB[i,m] = gam * max(B[i, m] - SB[i, m], SB[i, m] - B[i, m])
                    TC[i, m] = gam * max(C[i, m] - SC[i, m], SC[i, m] - C[i, m])
                    SA[i, m] = A[i, m]
                    SB[i, m] = B[i, m]
                    SC[i, m] = C[i, m]
                    A[i, m] = A[i, m] * AU[i, m] / AD[i, m] + TA[i,m]
                    B[i, m] = B[i,m] * BU[i, m] / BD[i, m] + TB[i,m]
                    C[i, m] = C[i, m] * CU[i, m] / CD[i, m] + TC[i,m]
            print('a,b,c')
            print(C)

            # 更新H,H_
            for m in range(e):
                for n in range(e):
                    for i in range(I):
                        for j in range(J):
                            if R[i, j] != 0:
                                seda = A[i, m] * A[j, n] + A[i, m] * B[j, n] + B[i, m] * A[j, n] + B[i, m] * B[j, n]
                                seda_ = A[i, m] * A[j, n] + A[i, m] * C[j, n] + C[i, m] * A[j, n] + C[i, m] * C[j, n]
                                HU[m, n] = HU[m, n] + R[i, j] * seda
                                HD[m, n] = HD[m,n] + (P[0,i] + P[0,j] + o[i, j]) * seda + lam * H[m, n]
                                H_U[m, n] = H_U[m,n] + beta * R_[i, j] * seda_
                                H_D[m, n] = H_D[m,n] + beta * (Q[0,i] + Q[0,j] + o_[i, j]) * seda_ + lam * H_[m, n]
                    TH[m,n] = gam * max(H[m, n] - SH[m, n], SH[m, n] - H[m, n])
                    TH_[m,n] = gam * max(H_[m, n] - SH_[m, n], SH_[m, n] - H_[m, n])
                    SH[m, n] = H[m, n]
                    SH_[m, n] = H_[m, n]
                    H[m, n] = H[m, n] * HU[m, n] / HD[m, n] + TH[m,n]
                    H_[m, n] = H_[m, n] * H_U[m, n] / H_D[m, n] + TH_[m,n]
        print('h')
        print(H)

        # 更新P,Q
        for i in range(I):
            for j in range(J):
                if R[i,j] != 0:
                    PU[0, i] = PU[0, i] + R[i, j]
                    PD[0, i] = PD[0, i] + P[0, i] + P[0, j] + o[i, j] + lam * P[0, i]
                    QU[0, i] = QU[0, i] + beta * R_[i, j]
                    QD[0, i] = QD[0, i] + beta * (Q[0, i] + Q[0, j] + o_[i, j]) + lam * Q[0, i]

            TP[0, i] = gam * max(P[0, i] - SP[0, i], SP[0, i] - P[0, i])
            TQ[0, i] = gam * max(Q[0, i] - SQ[0, i], SQ[0, i] - Q[0, i])
            SP[0, i] = P[0, i]
            SQ[0, i] = Q[0, i]
            P[0, i] = P[0, i] * PU[0, i] / PD[0, i] + TP[0, i]
            Q[0, i] = Q[0, i] * QU[0, i] / QD[0, i] + TQ[0, i]
        print('p,q')
        print(Q)

        o = (A + B).dot(H).dot((A + B).T)
        o_ = (A + C).dot(H_).dot((A + C).T)

        o = map_to_5_scale(o)
        o_ = map_to_3_scale(o_)
        print(f'第{t}次：')
        print(o)

        k = 0
        # 计算评价指标——NMAE、RMSE
        for i in range(I):
            for j in range(J):
                if R[i, j] != 0:
                    k = k + 1
                    err_nmae = err_nmae + abs(R[i, j] - o[i, j])
                    err_rmse = err_rmse + (R[i, j] - o[i, j]) ** 2

        Error_NMAE.append(err_nmae / k)
        Error_RMSE.append(math.sqrt(err_rmse / k))

        t = t + 1
    return Error_NMAE,Error_RMSE

NMAE,RMSE = NLF(R, R_, o, o_, I, J, e, A, B, C, P, Q, H, H_)
x = [i for i in range(999)]
print(RMSE)
plt.plot(x,RMSE)

plt.title('Method_Five')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.grid()

plt.show()

metric = pd.DataFrame([NMAE, RMSE], index=['NMAE', 'RMSE'])
metric = metric.T

print(metric)

metric.to_csv('metric/M5.csv', index=False)
