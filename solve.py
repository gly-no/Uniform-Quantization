import torch
import torch.nn as nn
from scipy.optimize import minimize, root
import numpy as np
import math
from model import LeNet
from model import FCnet
from numpy import *
from sympy import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from solve_test import train as solution
from solve_test import sci_solve, suf_st_beta, suf_st_normal, suf_st_kumaraswamy
import scipy.stats as st
from scipy import integrate
import pdf_cdf_plot as pcplot


def W_Quan_mulaw(model,M,t):
    #mapping to uniform
    ### 先归一化
    a1 = np.abs(model.F1.weight.data)
    a1 = a1.max()
    a2 = np.abs(model.F2.weight.data)
    a2 = a2.max()
    a3 = np.abs(model.OUT.weight.data)
    a3 = a3.max()
    w1_r = model.F1.weight.data
    w2_r = model.F2.weight.data
    w3_r = model.OUT.weight.data

    # w1_r = model.F1.weight.data/a1
    # n = len(w1_r)
    # m = len(w1_r[0])
    # for i in range(n):
    #     for j in range(m):
    #         if w1_r[i][j] < 0:
    #             w1_r[i][j] = -w1_r[i][j]

    w1 = (model.F1.weight.data/a1+1)/2  #移到(0,1)
    w2 = (model.F2.weight.data/a2+1)/2
    w3 = (model.OUT.weight.data/a3+1)/2

    ###解参数and mapping

    #mu-law
    mu1 = solve_params(w1)
    mu2 = solve_params(w2)
    mu3 = solve_params(w3)

    w1_u = np.log(1+mu1*w1.numpy())/np.log(1+mu1)
    w2_u = np.log(1+mu2*w2.numpy())/np.log(1+mu2)
    w3_u = np.log(1+mu3*w3.numpy())/np.log(1+mu3)
    
    # plot_hist(w1_r.numpy(), w1_u, M, t)
    # #beta-law
    # [al1,b1] = solve_params(w1)
    # [al2,b2] = solve_params(w2)
    # [al3,b3] = solve_params(w3)

    # tt = symbols('tt')
    # func1 = tt**(al1-1)*(1-tt)**(b1-1)
    # w1_u = integrate(func1, (tt, 0, w1))

    # func2 = math.pow(tt,al2-1)*math.pow(1-tt,b2-1)
    # w2_u = integrate(func2, (tt, 0, w2))

    # func3 = math.pow(tt,al3-1)*math.pow(1-tt,b3-1)
    # w3_u = integrate(func3, (tt, 0, w3))
    
    #uniform quantization
    w1_q = np.round((math.pow(2,M)-1)*w1_u)/(math.pow(2,M)-1)
    w2_q = np.round((math.pow(2,M)-1)*w2_u)/(math.pow(2,M)-1)
    w3_q = np.round((math.pow(2,M)-1)*w3_u)/(math.pow(2,M)-1)

    #inverse mapping

    # mu-law
    w1_Q = (2*(np.power(1+mu1,w1_q)-1)/mu1-1)*a1.numpy()
    w2_Q = (2*(np.power(1+mu2,w2_q)-1)/mu2-1)*a2.numpy()
    w3_Q = (2*(np.power(1+mu3,w3_q)-1)/mu3-1)*a3.numpy()


    #beta-law

    # if t % 10 == 9:
    plot_hist(w1.numpy(), w1_u, M, t)


    w1_Q = torch.from_numpy(w1_Q)
    w2_Q = torch.from_numpy(w2_Q)
    w3_Q = torch.from_numpy(w3_Q)

    model.F1.weight.data = w1_Q.to(torch.float)
    model.F2.weight.data = w2_Q.to(torch.float)
    model.OUT.weight.data = w3_Q.to(torch.float)

    # model.F1.weight.data = torch.zeros(100,784)
    # model.F2.weight.data = torch.zeros(20,100)
    # model.OUT.weight.data = torch.zeros(10,20)
    
    #grad

    w1_grad = np.power(1+mu1, w1_q)/(1+mu1*w1_q)
    w2_grad = np.power(1+mu2, w2_q)/(1+mu2*w2_q)
    w3_grad = np.power(1+mu3, w3_q)/(1+mu3*w3_q)

    # g1 = model.F1.weight.grad
    # g2 = model.F2.weight.grad
    # g3 = model.OUT.weight.grad
    
    # w1_grad = torch.from_numpy(w1_grad*g1)
    # w2_grad = torch.from_numpy(w2_grad*g2)
    # w3_grad = torch.from_numpy(w3_grad*g3)

    return model, w1_grad, w2_grad, w3_grad

def W_Quan_betalaw(w1, M, train_or_not = True, a1 = None, b1 = None, a2 = None, b2 = None):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # w1 = model.F2.weight.data
    w1 = w1.cpu().numpy()
    # w1.to(device)
    # print(w1.device)
    b = np.ceil(np.abs(w1).max())
    w11 = w1 - b - 1
    ww1 = np.exp(w11/10)
    # a2 = np.abs(model.F2.weight.data)
    # a3 = np.abs(model.OUT.weight.data)

    ####mapping
    if train_or_not == True:
        # a11,b11 = solution(ww1, 10000)
        a11, b11 = sci_solve(ww1)
        # a11 = a11.numpy()
        # b11 = b11.numpy()
        # print(a11, b11)
    else:
        a11 = a1
        b11 = b1
    # a11 = 7.9541
    # b11 = 2.2788
    # column1 = ww1.shape[0]
    # row1 = ww1.shape[1]
    # ff = lambda x: math.pow(x, a11-1)*math.pow(1-x, b11-1)
    # cdf1 = np.zeros((column1,row1))
    # gg1 = math.gamma(a11+b11)/(math.gamma(a11)*math.gamma(b11))

    # for i in range(column1):
    #     for j in range(row1):
    #         w_n, evv = integrate.quad(ff, 0, ww1[i][j])
    #         cdf1[i][j] = gg1 * w_n

    cdf1 = st.beta.cdf(ww1, a11, b11)

    if train_or_not == True:
        a12,b12 = solution(cdf1, 10000)
        a12 = a12.numpy()
        b12 = b12.numpy()
    else:
        a12 = a2
        b12 = b2
    # a12 = 13.8588
    # b12 = 11.3511

    # ff = lambda x: math.pow(x, a12-1)*math.pow(1-x, b12-1)
    # cdf2 = np.zeros((column1,row1))
    # gg2 = math.gamma(a12+b12)/(math.gamma(a12)*math.gamma(b12))
    # for i in range(column1):
    #     for j in range(row1):
    #         w_n, evv = integrate.quad(ff, 0, cdf1[i][j])
    #         cdf2[i][j] = gg2 * w_n
    
    cdf2 = st.beta.cdf(cdf1, a12, b12)

    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf2)/(math.pow(2,M)-1)

    ###inverse mapping
    ss1 = 1 - w_q
    ww1_Q_1 = st.beta.isf(ss1, a12, b12)
    ww1_Q_1 = ww1_Q_1.clip(cdf1.min(), cdf1.max())
    ss2 = 1 - ww1_Q_1
    ww1_Q_2 = st.beta.isf(ss2, a11, b11)
    ww1_Q_2 = ww1_Q_2.clip(ww1.min(), ww1.max())
    ww1_Q = np.log(ww1_Q_2)*10 + b + 1
    ww1_Q = ww1_Q.clip(-b, b)
    # model.F2.weight.data = torch.from_numpy(ww1_Q).to(torch.float32)
    # model.F1.weight.data

    
    ###calculate grad
    # grad1 = np.exp(w1)/ ww1_Q_2 *np.power(ww1/ww1_Q_1, a11-1)*np.power((1-ww1)/(1-ww1_Q_1), b11-1)*np.power(cdf1/cdf2, a12-1)*np.power((1-ww1)/(1-cdf2), b12-1)
    g1 = ww1 / ww1_Q_2
    g2 = np.power(ww1/ww1_Q_2, a11-1)
    g3 = np.power((1-ww1)/(1-ww1_Q_2), b11-1)
    g4 = np.power(cdf1/ww1_Q_1, a12-1)
    g5 = np.power((1-ww1)/(1-ww1_Q_1), b12-1)

    grad1 = g1*g2*g3*g4*g5


    # model.to(device)
    return torch.from_numpy(ww1_Q).to(torch.float32), grad1, a11, b11, a12, b12

def W_Quan_betalaw_new(w, M, train_or_not = True, a1 = None, b1 = None):
    # w_n = W_normalization(w)
    w1 = w.cpu().numpy()
    b = np.ceil(10*np.abs(w1).max())
    ww1 = (10*w1/b + 1)/2 
    lb = np.exp(-5)
    rb = np.exp(-0.01)
    ####mapping
    if train_or_not == True:
        a11, b11 = suf_st_beta(ww1)
        a11 = a11.numpy()
        b11 = b11.numpy()
    else:
        a11 = a1
        b11 = b1

    cdf1 = st.beta.cdf(ww1, a11, b11)
    
    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)
    # ma = cdf1.max()
    # mi = cdf1.min()
    # w_q = (np.ceil(math.pow(2,M)*cdf1.clip(0.0000001, 1))*2-1)/math.pow(2,M+1)
    # w_qq = w_q*(ma - mi) + mi

    '''creat look-up table'''
    tab = np.power(2, M) - 1
    dic = {}
    for i in range(tab + 1):
        dic[i/tab] = st.beta.isf(1-i/tab, a11, b11)

    ###inverse mapping
    # ss1 = 1 - w_q
    # ww1_Q_1 = st.beta.isf(ss1, a11, b11)
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    shape = w_q.shape
    ww1_Q_1 = np.zeros(shape)
    for idx, data in np.ndenumerate(w_q):
        ww1_Q_1[idx] = dic[data]
    ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    # ww1_Q_1 = ww1_Q_1.clip(lb, rb)
    ww1_Q = (ww1_Q_1 * 2 - 1) * b / 10
    ww1_Q = ww1_Q.clip(-b, b)

    # plt.clf()
    # plt.hist(w.cpu().reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'orignal',density= True)
    # # plt.hist(w_n.cpu().reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'normalzation',density= True) 
    # plt.hist(ww1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'normalization',density= True) 
    # plt.hist(cdf1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uniform',density= True)
    # # plt.hist(w_q.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uni_quan',density= True) 
    # # plt.hist(ww1_Q.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'quantization',density= True) 
    # plt.legend(loc = 'upper right')
    # plt.savefig("plot/normalization")
    # plt.show()
    
    g1 = np.power(ww1/ww1_Q_1, a11-1)
    g2 = np.power((1-ww1)/(1-ww1_Q_1), b11-1)
    grad1 = g1*g2
    grad1 = grad1.clip(-1000000,1000000)
    return torch.from_numpy(ww1_Q).to(torch.float32), grad1, a11, b11

def W_Quan_Kumaraswamy(w1, M, train_or_not = True, a1 = None, b1 = None):
    ###numpy, can be transfor to tensor
    w1 = w1.cpu().numpy()
    b = np.ceil(10*np.abs(w1).max())
    ww1 = (10*w1/b + 1)/2 
    lb = np.exp(-5)
    rb = np.exp(-0.01)
    ####mapping
    if train_or_not == True:
        a11, b11 = suf_st_kumaraswamy(ww1)
        # a11 = np.array(a11)
        # b11 = np.array(b11)
    else:
        a11 = a1
        b11 = b1

    cdf1 = 1 - np.power((1 - np.power(ww1, a11)),b11)
    cdf1 = np.float64(cdf1)
    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)
    # ma = cdf1.max()
    # mi = cdf1.min()
    # w_q = (np.ceil(math.pow(2,M)*cdf1.clip(0.0000001, 1))*2-1)/math.pow(2,M+1)
    # w_qq = w_q*(ma - mi) + mi

    '''creat look-up table'''
    tab = np.power(2, M) - 1
    dic = {}
    for i in range(tab + 1):
        dic[i/tab] = np.power(1 - np.power(1 - i/tab, 1/b11), 1/a11)

    ###inverse mapping
    # ss1 = 1 - w_q
    # ww1_Q_1 = st.beta.isf(ss1, a11, b11)
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    shape = w_q.shape
    ww1_Q_1 = np.zeros(shape)
    for idx, data in np.ndenumerate(w_q):
        ww1_Q_1[idx] = dic[data]

    ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    # ww1_Q_1 = ww1_Q_1.clip(lb, rb)
    ww1_Q = (ww1_Q_1 * 2 - 1) * b / 10
    ww1_Q = ww1_Q.clip(-b, b)

    
    # plt.hist(w1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'original',density= True) 
    plt.hist(ww1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'normalization',density= True) 
    plt.hist(cdf1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uniform_1',density= True)
    plt.hist(w_q.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uni_quan',density= True) 
    # plt.hist(ww1_Q.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'quantization',density= True) 
    pcplot.kumaraswamy_cdf_plot(a11,b11)
    pcplot.kumaraswamy_pdf_plot(a11,b11)
    plt.legend(loc = 'upper right')
    plt.show()
    
    g1 = np.power(ww1/ww1_Q_1, a11-1)
    g2 = np.power((1-np.power(ww1,a11))/(1-np.power(ww1_Q_1,a11)), b11-1)
    grad1 = g1*g2
    grad1 = grad1.clip(-1000000,1000000)

    return torch.from_numpy(ww1_Q).to(torch.float32), grad1, a11, b11

def W_Quan_normal(w1, M, train_or_not = True, a1 = None, b1 = None):
    # w1 = w1.cpu().numpy()
    # b = torch.ceil(torch.abs(w1).max())
    # ww1 = (w1/b + 1)/2 
    # lb = np.exp(-7)
    # rb = np.exp(-0.001)
    ####mapping
    if train_or_not == True:
        a11, b11 = suf_st_normal(w1)
        # a11 = a11.numpy()
        # b11 = b11.numpy()
    else:
        a11 = a1
        b11 = b1
    
    c1 = torch.distributions.normal.Normal(a11, torch.sqrt(b11))
    cdf1 = torch.distributions.normal.Normal.cdf(c1, w1)
    ma = torch.max(cdf1)
    mi = torch.min(cdf1)
    ###uniform quantization
    # w_q = torch.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)
    w_q = (torch.ceil(math.pow(2,M)*cdf1.clip(0.0000001, 1))*2-1)/math.pow(2,M+1)
    w_qq = w_q*(ma - mi) + mi
    ###inverse mapping
    w1_Q = torch.distributions.normal.Normal.icdf(c1,w_qq)
    w1_Q = w1_Q.clip(a11-3*torch.sqrt(b11), a11+3*torch.sqrt(b11))
    
    nu_wq = w_q.numpy()
    ori = w1.numpy()
    ori_quan = w_qq.numpy()
    Nor = cdf1.numpy()
    quan = w1_Q.numpy()

    plt.hist(ori.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'original',density= True) 
    plt.hist(ori_quan.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uni_quan',density= True) 
    plt.hist(Nor.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uniform_1',density= True)
    plt.hist(quan.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'quantization',density= True) 
    plt.legend(loc = 'upper right')
    plt.show()
    

    g1 = torch.distributions.normal.Normal.log_prob(c1,w1)
    g2 = torch.distributions.normal.Normal.log_prob(c1,w1_Q)
    grad1 = torch.exp(g1)/torch.exp(g2)

    return w1_Q.to(torch.float32), grad1, a11, b11

def W_grad(model,grad):
    
    g1 = model.F1.weight.grad
    g2 = model.F2.weight.grad
    g3 = model.OUT.weight.grad
    
    # gg1= torch.from_numpy(grad)*g1
    # gg2 = torch.from_numpy(grad[1])*g2
    # gg3 = torch.from_numpy(grad[2])*g3
    # gg1 = grad*g1
    gg2 = grad*g2
    # gg3 = grad[2]*g3

    return  gg2

def W_normalization(w):
    e = 0.00001
    mean = torch.mean(w)
    var = torch.var(w)
    w = (w - mean)/(torch.sqrt(var)+e)

    return w
   


if __name__ == "__main__":
    # w1 = np.array([[0.2, 0.3, 0.8],[0.6,0.5,0.1]])
    # solve_para(w=w1)
    model = FCnet()
    out = W_Quan_betalaw(model=model, M=2)
    # gg = W_grad(model=out[0],grad=out[1:3])
    print(out[0])
    # model.F1.weight.data = w1