import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import torch
from mpl_toolkits.mplot3d import Axes3D

def beta_pdf_plot(a,b,la = 'pdf'):
    g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0, 1, 0.01)
    pdf = []
    for t in x:
        y_1 = g*np.power(t,a-1)*np.power(1-t,b-1)
        pdf.append(y_1)
    plt.plot(x, pdf,label = la)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.ylim(0, 5)
    # plt.legend()
    # plt.show()

def beta_cdf_plot(a,b, la='cdf'):
    g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0.01, 1, 0.01)
    ff = lambda x: math.pow(x, a-1)*math.pow(1-x, b-1)
    cdf = []
    for t in x:
        ww, ev = integrate.quad(ff, 0, t)
        y_1 = g*ww
        cdf.append(y_1)
    plt.plot(x, cdf, label = la)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()

def kumaraswamy_pdf_plot(a,b,la = 'pdf'):
    # g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0, 1, 0.01)
    pdf = []
    for t in x:
        y_1 = a*b*np.power(t,a-1)*np.power(1-np.power(t,a),b-1)
        pdf.append(y_1)
    plt.plot(x, pdf,label = la)
    plt.xlabel("x")
    plt.ylabel("y")

def kumaraswamy_cdf_plot(a,b, la='cdf'):
    # g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0, 1, 0.01)
    # ff = lambda x: math.pow(x, a-1)*math.pow(1-x, b-1)
    cdf = []
    for t in x:
        # ww, ev = integrate.quad(ff, 0, t)
        y_1 = 1 - np.power(1 - np.power(t, a), b)
        cdf.append(y_1)
    plt.plot(x, cdf, label = la)
    plt.xlabel("x")
    plt.ylabel("y")

def loss_plot(w):
   
    n = w.shape[0]*w.shape[1]
    alpha = np.arange(0.01, 2, 0.01)
    beta = np.arange(5, 8, 0.01)
    aa = []
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(alpha, beta)

  
    for i in alpha:
        for j in beta:
            ll = math.gamma(i + j)/(math.gamma(i)+math.gamma(j))
            aa.append(ll)

    aa = np.array(aa).reshape((-1,len(alpha)))
    

    loss = - np.log(aa) - ((X-1)*np.sum(np.log(w)) + (Y - 1)*np.sum(np.log(1-w)))/n

    ax.plot_surface(X,Y,loss,rstride=1,cstride = 1,cmap = 'rainbow')

# def pdf(w,a,b):
#     g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
#     p = g*np.power(w,a-1)*np.power(1-w,b-1)

    # return p


if __name__ == '__main__':
    # a = 0.3389
    # b = 6.9903
    a = 4.8915
    b = 7.0750
    # beta_cdf_plot(a,b)
    # beta_pdf_plot(a,b)
    w = np.random.normal(0, 1, size = (200, 200))
    w = np.clip(w, -4, 4) - 5
    w = np.exp(w)

    loss_plot(w)

    # plt.legend()
    plt.show()

