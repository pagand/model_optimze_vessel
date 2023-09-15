# pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor, cuda, device
from torch.nn import Linear
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns # for visualization
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns # for visualization
import numpy as np
import matplotlib.pylab as plt
from torch.utils import data
import pickle
import time



# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path, normalized = False, n_step=0):
        # load the csv file as a dataframe
        df = read_csv(path, header=0, index_col=0)
        # store the inputs and outputs
        if n_step > 0:
            df['OUT'] = df['OUT'].shift(-n_step)
            df.dropna(inplace=True)
        self.MinOut = df.iloc[:, -1].min()
        self.MaxOut = df.iloc[:, -1].max()

        if normalized:
            for column in df.columns[:]:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())


        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')


        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer

        self.hidden1 = Linear(n_inputs, 30)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(30, 20)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(20, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # # third hidden layer and output
        X = self.hidden3(X)
        return X

# prepare the dataset
def prepare_data(path, normalized):
    # load the dataset
    dataset = CSVDataset(path, normalized ,n_step= 5)# normalize input and output

    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    MinOut = dataset.MinOut
    MaxOut = dataset.MaxOut


    return train_dl, test_dl, MinOut, MaxOut


# train the model
def train_model(train_dl, model, device0):
    # define the optimization
    criterion = MSELoss()
    #optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.007)

    # enumerate epochs
    for epoch in range(1000): #1000
        # if epoch % 10 ==0  and epoch:
        #     print("Epoch %d completed" %epoch)
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # convert to device
            inputs = inputs.to(device0)
            targets = targets.to(device0)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # if i == 100 or i == 500 or i ==900:
            #   print ([yhat])
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model, device0):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # convert to device
        inputs = inputs.to(device0)
        targets = targets.to(device0)
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    return mean_squared_error(actuals, predictions), r2_score(actuals, predictions)


# make a class prediction for one row of data
def predict(row, model, device0):
    # convert row to data
    row = Tensor([row])
    # move to device
    inputs = row.to(device0)
    # make prediction
    yhat = model(inputs)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return yhat


def plot(y_test, y_pred, fig_name):
    plt.figure()
    ax1 = sns.distplot(y_test, color='r', hist=False, label='actual')
    ax2 = sns.distplot(y_pred,  color='b', hist=False, label='prediction', ax=ax1)
    plt.legend(fontsize=5)
    plt.savefig(fig_name)


def main():
    dict = {} # feature importance for different methods
    # Access GPU ( before in google colab try to change Runtime>change runtime time>hardware acceleator > GPU)
    device0 = device('cuda:0' if cuda.is_available() else 'cpu')
    print("current device is {}".format(device0))
    # prepare the data

    path = './data/in_out_NN.csv'
    normalize = True
    train_dl, test_dl, MinOut, MaxOut = prepare_data(path, normalized=normalize)

    print(len(train_dl.dataset), len(test_dl.dataset))
    # define the network
    model = MLP(10).to(device0) # number of inputs, run in the device
    # train the model
    print("start training ...")
    st = time.time()
    train_model(train_dl, model, device0)
    # train overal error
    mse, r2 = evaluate_model(train_dl, model, device0)
    et = time.time()
    scaler = (MaxOut-MinOut) if normalize else 1 # 1 if no normalization, otherwise should put to denormalize
    actual_mse = mse * (scaler)**2
    print('train error:: MSE: %.3f, RMSE: %.3f' % (actual_mse, sqrt(actual_mse)))
    print('r2:: MSE: %.3f' % (r2))
    # saving the results
    dict['mlp'] = {}
    dict['mlp']['train_time'] = et - st
    # evaluate the model
    print("start evaluation ...")
    st = time.time()
    mse, r2_test = evaluate_model(test_dl, model, device0)
    et = time.time()
    actual_mse = mse * (scaler)**2
    print('test error:: MSE: %.3f, RMSE: %.3f' % (actual_mse, sqrt(actual_mse)))
    # make a single prediction
    dict['mlp']['test_time'] = et - st
    dict['mlp']['R2'] = [r2, r2_test, sqrt(actual_mse)]
    #dict['mlp']['model'] = model
    # plotting the results
    y_test = np.zeros(len(test_dl.dataset))
    y_pred = np.zeros(len(test_dl.dataset))
    A = np.array(train_dl.dataset)

    for i in range(len(test_dl.dataset)):
        y_test[i] =  A[i][1]*scaler+ MinOut
        y_pred[i] = predict(A[i][0], model, device0) *scaler + MinOut


    plot(y_test, y_pred, "mlp.eps")

    total_params = sum(param.numel() for param in model.parameters())

    print(total_params)

    with open('saved_dictionary_mlp.pkl', 'wb') as f:
        pickle.dump([dict], f)



if __name__ == '__main__':
    main()