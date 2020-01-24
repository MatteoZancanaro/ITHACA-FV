import numpy as np
from files import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
import io

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

plt.rcParams['figure.dpi'] = 200
device = torch.device('cpu')
device
batch_size = 500

class NutDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class Net(nn.Module):
    def __init__(self,Nu,Nnut,Epochs = 10000, activation = "tanh", dropout = 0.0):
        super(Net, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "celu":
            self.activation = nn.CELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        else:
            raise ValueError("Activation function not known.")
        self.fc1 = nn.Linear(Nu+1, 64) 
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, Nnut)
        self.dropout = nn.Dropout(dropout)
        self.Nu = Nu
        self.Nnut = Nnut
        self.Epochs = Epochs
    def forward(self, x):            
        x = self.fc1(x)  
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc6(x)
        return x


    def read(self):
        NOffSnap = np.loadtxt("ITHACAoutput/Offline/snaps",dtype=int)
        NOnSnap = np.loadtxt("ITHACAoutput/checkOff/snaps",dtype=int)
        
        ## Read the coefficients train
        # U
        inp_np_train_U = np.load("ITHACAoutput/NN/coeffs/coeffL2UTrain.npy")
        # P
        inp_np_train_P = np.load("ITHACAoutput/NN/coeffs/coeffL2PTrain.npy")
        # Nut
        out_np_train = np.load("ITHACAoutput/NN/coeffs/coeffL2NutTrain.npy")
        # Read Angles from file train
        angles_train = np.loadtxt("angOff_mat.txt")
        #NOffSnap = np.load("train/NOffSnap.npy")
        angles_train_np = []
        # Fill the train angles
        for k,j in enumerate(NOffSnap):
            for i in range(j):
                angles_train_np.append(angles_train[k])
        angles_train_np = np.asarray(angles_train_np)
        
        # Read the coefficients test
        # U
        inp_np_test_U = np.load("ITHACAoutput/NN/coeffs/coeffL2UTest.npy")
        # P
        inp_np_test_P = np.load("ITHACAoutput/NN/coeffs/coeffL2PTest.npy")
        # Nut
        out_np_test = np.load("ITHACAoutput/NN/coeffs/coeffL2NutTest.npy")
        # Read Angles from file test
        angles_test = np.loadtxt("angOn_mat.txt")
        #NOnSnap = np.load("test/NOnSnap.npy")
        angles_test_np = []
        # Fill the train angles
        for k,j in enumerate(NOnSnap):
            for i in range(j):
                angles_test_np.append(angles_test[k])
        angles_test_np = np.asarray(angles_test_np)
        
        # Prepare dataset with and without angles
        self.inp_np_train_a = np.append(np.transpose(np.expand_dims(angles_train_np,axis=0)),inp_np_train_U[:,0:self.Nu], axis = 1)
        self.inp_np_test_a = np.append(np.transpose(np.expand_dims(angles_test_np,axis=0)),inp_np_test_U[:,0:self.Nu], axis = 1)
        self.inp_np_train_noa = inp_np_train_U[:,0:self.Nu]
        self.inp_np_test_noa = inp_np_test_U[:,0:self.Nu]
        self.out_np_train = out_np_train[:,0:self.Nnut]
        self.out_np_test = out_np_test[:,0:self.Nnut]

    def trainNet(self):
        epochs = self.Epochs
        scaling = {"scaler_inp": preprocessing.MinMaxScaler(),
                   "scaler_out": preprocessing.MinMaxScaler()}
        inp_np_train = scaling["scaler_inp"].fit_transform(self.inp_np_train_a)
        out_np_train = scaling["scaler_out"].fit_transform(self.out_np_train)
        inp_np_test = scaling["scaler_inp"].transform(self.inp_np_test_a)
        out_np_test = scaling["scaler_out"].transform(self.out_np_test)
        trainset = NutDataset(inp_np_train, out_np_train)
        testset = NutDataset(inp_np_test, out_np_test)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=inp_np_test.shape[0], shuffle=True)
        
        # Loss Functions
        loss_fn = torch.nn.MSELoss(reduction = "mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)
        
        tplot = []
        lossplottrain = []
        lossplottest = []
        for t in range(epochs):
            self.train()
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            batch_losses = []
            for inputs, labels in trainloader:
                inputs, labels =  inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            loss = np.mean(batch_losses)
            
            # evaluate accuracy on test set
            batch_test_losses = []
            self.eval()
            with torch.no_grad():
                for inputs_test, labels_test in testloader:
                    inputs_test, labels_test =  inputs_test.to(device), labels_test.to(device)
                    outputs_test = self.forward(inputs_test)
                    test_loss = loss_fn(outputs_test, labels_test)
                    batch_test_losses.append(test_loss.item())
                test_loss = np.mean(batch_test_losses)
            if t % 100 == 99:
                print(t, "loss on train" , loss)
                print(t, "loss on test" , test_loss)
                tplot.append(t)
                lossplottrain.append(loss)
                lossplottest.append(test_loss)
            scheduler.step(test_loss)


        self.t_plot = tplot
        self.lossplottrain = lossplottrain
        self.lossplottest = lossplottest
        self.scaling = scaling
        # self.t_plot, self.lossplottrain, self.lossplottest,self.model,self.scaling = self.Net1(self.inp_np_train_a, self.inp_np_test_a, self.out_np_train, self.out_np_test, 
        #                                                       self.Epochs, 1e-4, 1e-7, 500)
    def plot_loss(self):
        plt.plot(self.t_plot, self.lossplottrain, label="train")
        plt.plot(self.t_plot, self.lossplottest, label="test")
        plt.legend()
        plt.show()

    def save(self):
        m = torch.jit.script(self)
        np.save("ITHACAoutput/NN/minAnglesInp_"+str(self.Nu) + "_" +str(self.Nnut) + ".npy",self.scaling["scaler_inp"].min_[:,None])
        np.save("ITHACAoutput/NN/scaleAnglesInp_"+str(self.Nu) + "_" +str(self.Nnut) + ".npy",self.scaling["scaler_inp"].scale_[:,None])
        np.save("ITHACAoutput/NN/minOut_"+str(self.Nu) + "_" +str(self.Nnut) + ".npy",self.scaling["scaler_out"].min_[:,None])        
        np.save("ITHACAoutput/NN/scaleOut_"+str(self.Nu) + "_" +str(self.Nnut) + ".npy",self.scaling["scaler_out"].scale_[:,None])
        m.save("ITHACAoutput/NN/Net_"+str(self.Nu) + "_" +str(self.Nnut)+".pt")

# sed_variable("NmodesUproj", "system/ITHACAdict", 10)

# Netok = Net(10,10,500)
# Netok.read()
# Netok.train()
# Netok.plot_loss()
# Netok.save()