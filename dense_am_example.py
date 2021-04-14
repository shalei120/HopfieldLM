import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dsets
import predictive_coding as pc
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.utils import save_image
from torch.autograd import Variable

from Hyperparameters import args


# Parameters of the model
input_dim = 28*28
hidden_dim = 2048
output_dim = 10

# Number of iterations performed by the train and test inference, respectively.
num_iterations = 20
t_test = 20

# Parameters of the algorithm. 
p_lr = 0.00005
x_lr = 0.1
xlr_test = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Partial represents the number of training example per label used.
# So, partial=10 means that we have a dataset of 100 training points.
partial = 10
batch_size = partial*10


#################################################################################
# Now, we load the training set:
# If you want to test using MNIST, just write MNIST instead of FashionMNIST.
train_dataset = dsets.FashionMNIST(root='./data', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)

train_list = []
label_list = []

for i in range(output_dim):
    count = 0
    for j in range(len(train_dataset)):
        if  (int(train_dataset[j][1]) == i) and (count < partial):
            label_list.append(train_dataset[j][0].view(input_dim))
            train_list.append(torch.tensor([1]))
            count += 1

tensor_train = torch.stack(train_list)
tensor_label = torch.stack(label_list)

my_dataset =  TensorDataset(tensor_train,tensor_label)

train_loader = DataLoader(dataset = my_dataset,
                            batch_size=batch_size,
                            shuffle = True,
                            drop_last=True,
                            )


####################################################################################
# We have loaded the dataset. We now define the model:

class AML_Model(nn.Module):
    def __init__(self, mid_features: int, out_features: int) -> None:
        super(AML_Model, self).__init__()

        self.fc1 = nn.Linear(1, mid_features)
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.fc3 = nn.Linear(mid_features, out_features)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.PClayer1 = pc.PCLayer() #PC Layer is what allows us to run inference
        self.PClayer2 = pc.PCLayer()
        self.PClayer3 = pc.PCLayer()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.PClayer1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.PClayer2(out)
        out = self.fc3(out)
        out = self.PClayer3(out)
        return out

####################################################################################

model = AML_Model(hidden_dim,input_dim)#.to(args['device'])

# We define two training environment: One for training, and one for testing:
model_trainer = pc.PCTrainer(model, optimizer_x_kwargs={'lr': x_lr}, optimizer_p_kwargs={'lr': p_lr}, x_lr_discount=0.9, T = num_iterations)#, is_sample_x_once = True)
prediction_trainer = pc.PCTrainer(model, optimizer_x_kwargs={'lr': xlr_test}, optimizer_p_kwargs={'lr': 0},x_lr_discount=0.9, T = t_test)



# This function trains the model
def fit(model, dataloader):

    def train_call(t):#, trainer):
        model.PClayer3._x.data.copy_(lab)

    model.train()
    loss = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        lab = labels
        images = Variable(images.view(-1,1).to(args['device']))
        results = model_trainer.train_on_batch(images.float(),callback_after_t= train_call)
        loss += results['overall'][num_iterations-1]
    loss = loss/len(train_loader.dataset)
    return loss

# This function evaluates the model
def validate(model, dataloader):

    def prediction(t):#, trainer):
        model.PClayer3._x.data[:, :392].copy_(lab)

    loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        model.train()
        lab = labels[:,:392]
        images = Variable(images.view(-1,1).to(args['device']))
        results = prediction_trainer.train_on_batch(images.float(),callback_after_t=prediction)
        loss += results['overall'][t_test-1]
        reconstruction = model.PClayer3._x.data.cpu()
        if i ==  0:
            num_rows = 8
            both = torch.cat((labels.view(batch_size, 1,28,28)[:8], 
                                reconstruction.view(batch_size, 1,28,28)[:8]))
            save_image(both.cpu(), f"./HD={hidden_dim},T={num_iterations},xlr={x_lr},plr={p_lr},par={partial},T_Test={t_test}.png", nrow=num_rows)
    val_loss = loss/len(dataloader.dataset)
    return val_loss




train_loss = []
val_loss = []
train_epoch_loss = fit(model, train_loader)
val_loss = validate(model, train_loader)
train_loss.append(train_epoch_loss)
print("T = {}, T_Test = {}, x test = {}, HD = {}, xlr = {}, plr = {}, par = {} - Train Loss: {}, Val Loss: {}".format(num_iterations,t_test,xlr_test,hidden_dim,x_lr,p_lr,partial,train_epoch_loss,val_loss))