import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from scipy import integrate
# from solve import plot_hist
from pdf_cdf_plot import beta_pdf_plot
from pdf_cdf_plot import beta_cdf_plot
# from pdf_cdf_plot import loss_plot
import scipy.stats as st
import scipy.special as ss
from sympy import symbols, Eq, nsolve
from scipy.optimize import minimize, root, fsolve
import cvxpy as cp


def integ(a):
    f = lambda x1 : math.exp(-x1)*torch.pow(x1, a-1)
    f_diff = lambda x2 : math.exp(-x2)*torch.pow(x2, a-1)*torch.log(torch.tensor(x2))

    f_i, er = integrate.quad(f,0,np.inf)
    f_diff_i, er = integrate.quad(f_diff,0,np.inf)
    return f_diff_i / f_i

def train(w, epoch = 10000):
    w = torch.from_numpy(w)
    alpha = torch.tensor(2.0)
    beta = torch.tensor(6.0)
    # epoch = 20000
    best_loss = 10 ** 5
    lr = 0.01

    n = w.shape[0]*w.shape[1]
    w_inver = torch.clamp(1 - w, np.exp(-n), 1)

    fig = plt.figure()
    plt.ion()
    x,y,z = [],[],[]
    
    qq = 0
    for _ in range(epoch):
        count = 1

        '''
        目标方程
        '''
        loss = - torch.log(torch.tensor(math.gamma(alpha + beta)/(math.gamma(alpha)*math.gamma(beta)))) - ((alpha-1)*torch.sum(torch.log(w)) + (beta - 1)*torch.sum(torch.log(w_inver)))/n
        
        g1,g2,g3 = integ(alpha),integ(beta),integ(alpha + beta)
        aaaa = torch.sum(torch.log(w))/n
        alpha_grad = g1 - g3 - torch.sum(torch.log(w))/n
        beta_grad = g2 - g3 - torch.sum(torch.log(w_inver))/n
        
        if loss < best_loss:
            '''
            修改
            '''
            best_loss = loss

            
            best_alpha = alpha.data
            best_beta = beta.data
            

            fig.clf()
            x.append(copy.deepcopy(best_alpha.numpy()))
            y.append(copy.deepcopy(best_beta.numpy()))
            z.append(best_loss.detach().numpy())
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter3D(x, y, z, marker="o")
            ax.set(xlim = (-6,5), ylim = (0,11), zlim = (-5, -1))
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Beta")
            ax.set_zlabel("Loss")
            plt.pause(0.1)

            count = 0

        alpha = alpha - alpha_grad * lr
        beta = beta - beta_grad * lr
        print(best_loss,loss)
     

        if count == 1:
            qq += 1
        else:
            qq = 0

        if qq == 30:
            break

    '''
    输出变量值
    '''
    # print("mu:", best_mu)
    print("alpha:",best_alpha)
    print("beta:",best_beta)
    # print("theta:", best_theta / np.pi * 180)

    plt.ioff()
    plt.show()
    return best_alpha,best_beta

def sci_solve(w):
    e = math.exp(-10)
    w = torch.from_numpy(w)
    n = w.shape[0]*w.shape[1]
    w_inver = 1 - w
    fun = lambda x:- torch.log(torch.tensor(math.gamma(x[0] + x[1])/(math.gamma(x[0])*math.gamma(x[1])))) - ((x[0]-1)*torch.sum(torch.log(w)) + (x[1] - 1)*torch.sum(torch.log(w_inver)))/n
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - e},
            {'type': 'ineq', 'fun': lambda x: x[1] - e})
    res = minimize(fun, (4, 6), method = 'SLSQP', constraints = cons)
    print(res.success)

    return res.x

def cvx(w):
    alpha = cp.Variable()
    beta = cp.Variable()
    e = math.exp(-10)
    n = w.shape[0]*w.shape[1]
    # p1 = (alpha-1)*cp.sum(cp.log(w))/n
    # p2 = (beta - 1)*cp.sum(cp.log(1-w))/n
    prom = cp.Maximize(cp.loggamma(alpha + beta) - cp.loggamma(alpha) - cp.loggamma(beta) + (alpha-1)*cp.sum(cp.log(w))/n + (beta - 1)*cp.sum(cp.log(1-w))/n)
    # prom = cp.Minimize(alpha + beta)
    cons = [alpha >= e, beta >= e]
    solverr = cp.Problem(prom, cons)
    print(solverr.is_dcp())
    solverr.solve()
    print(solverr.status, solverr.value, alpha.value, beta.value)
    return alpha.value, beta.value

def suf_st_beta(w):
    w = torch.from_numpy(w)
    # n = w.shape[0]*w.shape[1]
    mean = torch.mean(w)
    # mean2 = torch.sum(w)/n
    var = torch.var(w)
    # print(mean,var)
    alpha = mean*((1-mean)*mean/var - 1)
    beta = (1-mean)*((1-mean)*mean/var - 1)
    # print(alpha, beta)
    return alpha, beta

def suf_st_normal(w):
    mean = torch.mean(w)
    var = torch.var(w)
    print(mean, var)
    return mean,var

def suf_st_kumaraswamy(w):
    w = torch.from_numpy(w)
    mean = torch.mean(w)
    var = torch.var(w)

    # x, y = symbols('x,y')
    # eqs = [Eq(y*ss.beta(1+1/x,y)-mean), Eq(y*ss.beta(1+2/x,y) - (y*ss.beta(1+1/x,y))**2 - var)]
    # a,b = nsolve(eqs, [x,y], [1,1])
    # def beta(x,y):
    #     beta = math.gamma(x)*math.gamma(y)/math.gamma(x+y)
    #     return beta

    def f(x):
        return [x[1]*ss.beta(1+1/x[0],x[1])-mean, x[1]*ss.beta(1+2/x[0],x[1]) - (x[1]*ss.beta(1+1/x[0],x[1]))**2 - var]

    a, b = fsolve(f,[1.0,1.0])
    # print(a, b)
    return a, b

if __name__ == '__main__':
    w = np.random.normal(0, 1, size = (200, 200))
    w1 = w.reshape(1,-1)[0]
    plt.hist(w1, bins= 100, alpha = 0.5, label = 'original', density=True)
    # plt.show()

    # ###权值处理
    # ww = np.clip(w, -4, 4) - 5
    # ww = np.exp(ww)

    ###权值处理
    b = np.ceil(np.max(np.abs(w)))
    # ww = (w / b + 2)/2
    # ww = w - b - 1
    # ww = np.exp((w - b - 1)/10)
    ww = (w / b + 1) / 2 

    w2 = ww.reshape(1,-1)[0]
    plt.hist(w2, bins= 100, alpha = 0.5, label= 'normalization',density= True)
    # plt.show()
    

    ###还原
    # a1,b1 = train(ww)
    # 8.5434, 7.0501

    # a1, b1 = cvx(ww)
    a1, b1 = suf_st(ww)

    # a1, b1 = sci_solve(ww)
    # 3.9957, 5.9943
    

    column = ww.shape[0]
    row = ww.shape[1]
    # a1 = 6.1078
    # b1 = 4.1230
    beta_pdf_plot(a1,b1,la='pdf_1')
    beta_cdf_plot(a1,b1,la='cdf_1')
    plt.legend(loc='upper right')
    # plt.show()


    # ff = lambda x: math.pow(x, a1-1)*math.pow(1-x, b1-1)
    # cdf = np.zeros((column,row))
    # gg1 = math.gamma(a1+b1)/(math.gamma(a1)*math.gamma(b1))

    # for i in range(column):
    #     for j in range(row):
    #         w_n, evv = integrate.quad(ff, 0, ww[i][j])
    #         cdf[i][j] = gg1 * w_n
    cdf = st.beta.cdf(ww, a1, b1)

    cdf1 = cdf.reshape(1,-1)[0]

    # plt.hist(cdf1,bins = 100, alpha = 0.5, label='uniform_1',density=True)
    # plt.legend(loc='upper right')
    # plt.show()

    # # a2,b2 = train(cdf)
    # a2, b2 = suf_st(cdf)
    # # a2 = 4.6915
    # # b2 = 4.3646
    
    # ff = lambda x: math.pow(x, a2-1)*math.pow(1-x, b2-1)
    # cdf2 = np.zeros((column,row))

    # # beta_pdf_plot(a2,b2,'pdf_2')
    # # beta_cdf_plot(a2,b2,'cdf_2'/)
    # gg2 = math.gamma(a2+b2)/(math.gamma(a2)*math.gamma(b2))
    # for i in range(column):
    #     for j in range(row):
    #         w_n, evv = integrate.quad(ff, 0, cdf[i][j])
    #         cdf2[i][j] = gg2 * w_n

    # cdf2 = cdf2.clip(0,1)
    # cdf3 = cdf2.reshape(1,-1)[0]
    # plt.hist(cdf3, bins = 100, alpha = 0.5, label='uniform_2',density=True)
    # plt.legend(loc='upper right')
    # plt.show()

    ###uniform quantization
    M = 8
    w_q = np.round((math.pow(2,M)-1)*cdf)/(math.pow(2,M)-1)
    # w_qqq = w_q.reshape(1,-1)[0]
    # plt.hist(w_qqq, bins = 100, alpha = 0.5, label='w_q', density=True)
    # plt.legend(loc='upper right')
    # plt.show()

    ###逆映射


    '''逆映射可能不对'''
    # ss1 = 1 - w_q
    # w_Q_1 = st.beta.isf(ss1, a2, b2)
    # ss2 = 1 - w_Q_1
    ss2 = 1 - w_q
    w_Q_2 = st.beta.isf(ss2, a1, b1)
    # w_Q_2 = w_Q_2.clip(math.exp(-10), w_Q_2.max())
    # w_Q = np.log(w_Q_2)*10 + b + 1
    w_Q = (w_Q_2 * 2 - 1) * b
    w3 = w_Q.reshape(1,-1)[0]
    w_Q = w_Q.clip(-b, b)
    # plt.hist(w_Q_1.reshape((1,-1))[0], bins = 100, alpha = 0.5, label = 'w_Q_1', density=True)
    # plt.hist(w_Q_2.reshape((1,-1))[0], bins = 100, alpha = 0.5, label = 'w_Q_2', density=True)
    plt.hist(w_Q.reshape((1,-1))[0], bins = 100, alpha = 0.5, label = 'w_Q', density=True)
    plt.legend(loc='upper right')
    plt.show()

    w3 = w_Q.reshape(1,-1)[0]
    # plt.hist(w3, bins = 16, alpha = 0.5, label = 'Quantization', density=True)
    # plt.legend(loc='upper right')
    # plt.show()
    w3=33

    ###嵌入进训练的代码！！！