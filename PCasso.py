import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, draw, pause
import datetime, math
from Hyperparameters import args
import argparse

from torchviz import make_dot
class PCasso(nn.Module):

    def __init__(self, hidden_size=None, input_size = args['embeddingSize']):

        super(PCasso, self).__init__()
        if hidden_size is None:
            self.hidden_size = [50, 100, 100]
        self.s_in = torch.ones([1],device=args['device'])
        # self.linear_1 = nn.Linear(1, self.hidden_size[0])
        # self.linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        # self.linear_3 = nn.Linear(self.hidden_size[1], input_size)
        # self.linear_1 = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(True)
        # ) # 4 4
        # self.linear_2 = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(True)
        # ) # 8 8
        # self.linear_3 = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(True)
        # )  # 16b16
        #
        # self.linear_4 = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 2, 2, 2, bias=False),
        #     nn.Tanh()
        # )
        self.linear_1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(0.2),
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        self.linear_4 = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def base_arch(self, s_out, bsz = None):
        # print(self.s_in.size())
        x1 = self.linear_1(self.s_in)
        # print(self.s_in.size(), x1.size())
        if self.first_time:
            self.mu1 = Parameter(x1.repeat([bsz,1]))
        x2 = self.linear_2(self.mu1)

        if self.first_time:
            self.mu2 = Parameter(x2)
        x3 = self.linear_3(self.mu2)

        if self.first_time:
            self.mu3 = Parameter(x3)
            self.first_time = False
        x4 = self.linear_4(self.mu3)

        # print(x4.size(), s_out.size()) .sum(3).sum(2).sum(1)
        error4 = 0.5 * ((x4 - s_out) ** 2).mean()
        error3 = 0.5 * ((x3 - self.mu3) ** 2).mean()
        error2 = 0.5 * ((x2 - self.mu2) ** 2).mean()
        error1 = 0.5 * ((x1 - self.mu1) ** 2).mean()
        E = error1 + error2 + error3 + error4
        return E

    def forward(self, input, fixed_mask = None, gamma = 0.5) :
        batch_size = input.size()[0]
        input = input.reshape(batch_size, -1)
        if fixed_mask is not None:
            fixed_mask = fixed_mask.reshape(batch_size, -1)
        torch.autograd.set_detect_anomaly(True)
        if fixed_mask is None:
            fixed_mask = torch.ones(input.size())
            # print(input[0])
        # self.mu1 = Parameter(torch.rand([batch_size,1, 4,4])).to(args['device'])
        # self.mu2 = Parameter(torch.rand([batch_size,1, 8,8])).to(args['device'])
        # self.mu3 = Parameter(torch.rand([batch_size,1, 16,16])).to(args['device'])
        self.mu1 = Parameter(torch.rand([batch_size,256])).to(args['device'])
        self.mu2 = Parameter(torch.rand([batch_size,512])).to(args['device'])
        self.mu3 = Parameter(torch.rand([batch_size,1024])).to(args['device'])
        s_out = Parameter(input).to(args['device'])
        # print(s_out.size())
        # self.mu3 = torch.rand([batch_size,hidden_size[2]])



        old_mu1, old_mu2, old_mu3 = copy.deepcopy(self.mu1), copy.deepcopy(self.mu2), copy.deepcopy(self.mu3)
        old_sout = copy.deepcopy(s_out)
        self.first_time = True
        while True:
            # print('wk')

            E = self.base_arch(s_out, batch_size)
            self.mu3.requires_grad = True
            # self.mu3.grad.data.zero_()
            dx3 = torch.autograd.grad(E, self.mu3, retain_graph=True)
            self.mu3.requires_grad = False
            # print(dx2[0])
            with torch.no_grad():
                self.mu3= Parameter(self.mu3.data - gamma * dx3[0])

            self.mu2.requires_grad = True
            dx2 = torch.autograd.grad(E, self.mu2, retain_graph=True)
            self.mu2.requires_grad = False
            # print(dx2[0])
            with torch.no_grad():
                self.mu2 = Parameter(self.mu2.data - gamma * dx2[0])

            self.mu1.requires_grad = True
            dx1 = torch.autograd.grad(E, self.mu1, retain_graph=True)
            self.mu1.requires_grad = False

            with torch.no_grad():
                self.mu1 = Parameter(self.mu1.data - gamma * dx1[0])

            s_out.requires_grad = True
            dx_so = torch.autograd.grad(E, s_out, retain_graph=True)
            s_out.requires_grad = False

            with torch.no_grad():
                s_out = Parameter(s_out.data - gamma * dx_so[0] * (1-fixed_mask))

            # error1 = ((x1 - self.mu1)**2).sum()
            dims = [1]# [1,2,3]
            with torch.no_grad():
                loss1 = (old_mu1 - self.mu1).norm(p=2, dim=dims).max(axis=0)[0]
                loss2 = (old_mu2 - self.mu2).norm(p=2, dim=dims).max(axis=0)[0]
                loss3 = (old_mu3 - self.mu3).norm(p=2, dim=dims).max(axis=0)[0]
                loss4 = (old_sout - s_out).norm(p=2, dim=dims).max(axis=0)[0]
                # print(loss1, loss2, loss3, loss4)
                if loss1 < 1e-3 and loss2 < 1e-3 and loss3 < 1e-3 and loss4 < 1e-3:
                    break

            old_mu1, old_mu2, old_mu3 = copy.deepcopy(self.mu1), copy.deepcopy(self.mu2), copy.deepcopy(self.mu3)
            old_sout = copy.deepcopy(s_out)


        E = self.base_arch(s_out)
        return E, s_out.reshape(batch_size,1,28,28)

def Image_test():
    data = np.load('./mnist.npz')
    train_X, train_y, test_X, test_y = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    batch_size = 2
    data_n = 2 #train_y.shape[0]

    model = PCasso(input_size=28*28).to(args['device'])
    # print(train_X.shape)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-3, amsgrad=True)

    old_E = -1
    # for epoch in range(10000):
    #     for i in range(0, data_n, batch_size):
    #         optimizer.zero_grad()
    #         X = 1.0*train_X[i:min(i + batch_size, data_n),:,:] / 255
    #         bsz = X.shape[0]
    #         # Y = train_y[i:min(i + batch_size, data_n)]
    #         X = torch.FloatTensor(X).unsqueeze(1)
    #         E, s_out = model(X)
    #         loss_mean = E
    #
    #             # Reward = loss_mean.data
    #         loss_mean.backward(retain_graph=True)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
    #         optimizer.step()
    #     if np.abs(old_E - E.data) < 0.0000001:
    #         break
    #     old_E = E.data
    #     print('Epoch ', epoch, 'E=', E)
    #
    # torch.save(model, './pcmodel.mdl')

    model = torch.load('./pcmodel.mdl')

    sample = 0
    fixed_mask = torch.ones([1,1,28,28])
    fixed_mask[:,:, 14:,:] = 0
    data = train_X[sample] * fixed_mask.numpy()[0,0,:,:]


    model.eval()
    # print(train_X[sample].shape)
    input = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)

    fg = figure()
    ax = fg.gca()
    im = ax.imshow(data, cmap='gray')
    # im.set_data(data)
    #
    # draw()
    s_out = input / 255
    for i in range(100):
        e, s_out = model(s_out, fixed_mask = fixed_mask)
        # print(s_out)
        pixel = (s_out.reshape([28,28]) * 255).long()
        im.set_data(pixel)
        ax.set_title(str(i))
        draw(), pause(0.01)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g')

    cmdargs = parser.parse_args()
    if cmdargs.gpu is None:
        usegpu = False
        args['device'] = 'cpu'
    else:
        usegpu = True
        args['device'] = 'cuda:' + str(cmdargs.gpu)
    Image_test()










