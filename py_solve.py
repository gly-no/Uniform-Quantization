import torch
import numpy as np
from torch import nn
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

'''
变量
'''
param = {
    # "mu":10.1,
    # "i":20.2
    "alpha":1.0,
    "beta":1.0
}

'''
一般变量
'''
cfg_constant = {

}

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        for key,value in param.items():
            setattr(self,key,nn.Parameter(torch.tensor(value)))
        for key,value in cfg_constant.items():
            setattr(self,key,nn.Parameter(torch.tensor(value)))
    # def rt_cfg(self):
    #     # return *param.keys(), *cfg_constant.keys()
    #     return param.keys()
    def forward(self):
        # return [getattr(self,tmp) for tmp in self.rt_cfg()]
        return [getattr(self,tmp) for tmp in param.keys()]

def train(w):
    net = model()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    epoch = 10
    best_loss = 10 ** 5


    ###对权值进行一些预处理
    w = w.clip(-1, 1) - 1
    # bb = np.sum(w)
    w = torch.from_numpy(np.exp(w))
    # aa = torch.sum(torch.log(w))
    # aa1 = torch.sum(torch.log(1 - w))
    # w = torch.from_numpy(w)
    n = w.shape[0]*w.shape[1]
    w_inver = torch.clamp(1 - w, np.exp(-n), np.exp(-2))
    
    

    # w = torch.reshape(w,(1,-1))
    
    # s1 = torch.cumprod(w , 1)[0, n-1]
    # s2 = torch.cumprod(1-w , 1)[0, n-1]

    '''
    变量最佳值
    '''
    best_mu = None
    best_theta = None

    # best_alpha = None
    # best_beta = None

    fig = plt.figure()
    plt.ion()
    x,y,z = [],[],[]

    # print(type(mu))
    for _ in range(epoch):
        '''
        变量值 
        '''
        # mu, i = net()
        alpha, beta = net()
        
        '''
        目标方程
        '''
        # loss = (torch.sin(i) + torch.sin(theta) - 1.2) ** 2 + (theta - i - torch.tensor(33 / 180 * np.pi)) ** 2
        # loss = np.sum(np.log(1+mu*w))/n - np.log(mu/(np.log(1+mu))) - e

        # loss = torch.sum(-(math.gamma(alpha+beta)*torch.pow(w,alpha-1)*torch.pow(1-w,beta-1))/(math.gamma(alpha)*math.gamma(beta)))/n

        loss = - torch.log10(torch.tensor(math.gamma(alpha + beta)/(math.gamma(alpha)+math.gamma(beta)))) - ((alpha-1)*torch.sum(torch.log10(w)) + (beta - 1)*torch.sum(torch.log10(w_inver)))/n
        # loss = - torch.log10(torch.tensor(math.gamma(alpha + beta)/(math.gamma(alpha)+math.gamma(beta)))) - ((alpha-1)*s1 + (beta - 1)*s2)/n
        # loss = ((alpha-1)*torch.sum(torch.log10(w)))/n


        print(loss)
        # loss = loss.detach.numpy()
        # loss = torch.sum(torch.log10(1+mu*w))/n - torch.log10(mu/(torch.log2(1+mu)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            '''
            修改
            '''
            best_loss = loss

            # best_mu = mu
            # best_theta = theta
            best_alpha = alpha.data
            best_beta = beta.data
            # print(best_alpha,best_beta)

            fig.clf()
            x.append(copy.deepcopy(best_alpha.numpy()))
            y.append(copy.deepcopy(best_beta.numpy()))
            z.append(best_loss.detach().numpy())
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter3D(x, y, z, marker="o")
            # ax.set(xlim = (-20,20), ylim = (-20,20), zlim = (-5,5))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.pause(0.5)

        print(best_loss)
        ####plot  loss


    '''
    输出变量值
    '''
    # print("mu:", best_mu)
    print("alpha:",best_alpha)
    print("beta:",best_beta)
    # print("theta:", best_theta / np.pi * 180)

    plt.ioff()
    plt.show()





if __name__ == '__main__':
    # w = np.random.normal(0, 1, size = (20, 20))
    # a = np.abs(w).max()
    # w = (w/a+1)/2
    w = np.random.normal(0, 1, size = (20, 20))
    # w = np.random.uniform(0, 1, size = (2, 2))
    # w = np.array([[0.1, 0.8],[0.4, 0.3]])
    # print(w)
    train(w)
