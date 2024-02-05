import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from baseline.dataset import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_standardize_data(df, dataset_name):

    # Define a data object to get the labels
    data_obj = dataset.Dataset(dataset_name)
    labels = data_obj.cfg.labels

    # Remove NaN rows
    df = df.dropna()

    # Initializing an encoder
    encoder = preprocessing.LabelEncoder()

    # Separate numerical & categorical data
    df3 = df.select_dtypes(include=['object']).copy()

    # Extract the numerical features
    if not df3.empty:
        df_new = df.select_dtypes(exclude=['object']).copy()
        # Encode the categorical data
        df3 = df3.apply(encoder.fit_transform)
        # Merge the different parts of the data set
        final_df = df_new.merge(df3, left_index=True, right_index=True)
    else:
        df_new = df.copy()  # there are no categorical features
        final_df = df_new

    # Rearrange the features and labels
    final_df = final_df[[col for col in final_df.columns if col != labels] + [labels]]

    # Randomly split the test ans train sets
    X_train, X_test = train_test_split(final_df, test_size=0.3, random_state=42)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler, encoder


class DataBuilder(Dataset):
    def __init__(self, data_df, dataset_name, train=True):
        self.X_train, self.X_test, self.standardizer, self.encoder = load_and_standardize_data(data_df, dataset_name)
        if train:
            self.x = self.X_train
            self.len = self.x.shape[0]
        else:
            self.x = self.X_test
            self.len = self.x.shape[0]
        del self.X_train
        del self.X_test

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


class Autoencoder(nn.Module):
    def __init__(self, D_in, H=50, H2=12, latent_dim=2):

        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        #         # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train(epoch, model, loss_mse, trainloader, optimizer, train_losses):
    model.train()

    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 1 == 0:
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))


def test(epoch, model, loss_mse, testloader, optimizer, test_losses):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(testloader):
            print(batch_idx)
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if epoch % 1 == 0:
                print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(testloader.dataset)))
            test_losses.append(test_loss / len(testloader.dataset))
