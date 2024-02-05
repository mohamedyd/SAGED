import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from baseline.setup.data_augmentation.autoencoder import DataBuilder, Autoencoder, device, customLoss, train, test
from baseline.setup.detectors.detect_method import DATA_PATH
from baseline.dataset import dataset


def run_vae(data_df, dataset_name, epochs=20, nb_samples=6000):
    """
    Generate new data using variational autoencoder
    """

    # Define a data object to get the labels
    data_obj = dataset.Dataset(dataset_name)

    # Get list of column names
    cols = data_df.columns
    target = data_obj.cfg.labels
    ml_task = data_obj.cfg.ml_task

    # Load and standardize train and test data
    train_set = DataBuilder(data_df, dataset_name, train=True)
    test_set = DataBuilder(data_df, dataset_name, train=False)

    # Create a data loader
    trainloader= DataLoader(dataset=train_set, batch_size=5, drop_last=True)
    testloader= DataLoader(dataset=test_set, batch_size=5, drop_last=True)

    # Prepare the configurations
    D_in = trainloader.dataset.x.shape[1]
    # Number of units in the first hidden layer
    H = 50
    # Number of units in the second hidden layer
    H2 = 12

    model = Autoencoder(D_in, H, H2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = customLoss()
    epochs = epochs
    log_interval = 50
    val_losses = []
    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        train(epoch,model,loss_mse, trainloader, optimizer,train_losses)
        test(epoch,model,loss_mse, testloader, optimizer,test_losses)

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

    # Invoke the standard scaler and the encoder used while preparing the input data
    scaler = trainloader.dataset.standardizer
    encoder = trainloader.dataset.encoder

    sigma = torch.exp(logvar/2)
    nb_samples = nb_samples
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([nb_samples]))

    with torch.no_grad():
        # Use the latent factors z to generate new data
        pred = model.decode(z).cpu().numpy()

    # Restore the original ranges of the input data
    fake_data = scaler.inverse_transform(pred)

    df_fake = pd.DataFrame(fake_data, columns=cols)
    if ml_task == 'binary_classification':
        df_fake[target] = np.round(df_fake[target]).astype(int)
        df_fake[target] = np.where(df_fake[target]<0, 0, df_fake[target])
        df_fake[target] = np.where(df_fake[target] > 1, 1, df_fake[target])

    return df_fake


if __name__ == "__main__":
    # Get the data path
    dataset_name = 'beers'

    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

    # Load the data
    data_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)

    df_fake = run_vae(data_df, dataset_name, epochs=20, nb_samples=4000)
    print(df_fake.shape)