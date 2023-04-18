import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import random
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from typing import *
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable

class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x
class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_rate=0,device='cpu'):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,batch_first=True)
        self.dropout=VariationalDropout(dropout_rate,batch_first=True)
        self.device = device
        self.out = nn.Linear(hidden_dim, output_dim, bias=True)

        #Orthogonal weight initialisation
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(p)
            elif 'bias' in name:
                nn.init.constant(p, 0)

    def forward(self, x):
        x = self.dropout(x)
        batch_size = x.size()[0]
        h0 = Variable(torch.zeros([1,batch_size, self.hidden_dim]), requires_grad=False).to(device=self.device)
        c0 = Variable(torch.zeros([1,batch_size, self.hidden_dim]), requires_grad=False).to(device=self.device)
        fx, _ = self.lstm.forward(x, (h0, c0))
        y=self.out(fx[:,-1])
        return y
####the start of the function
depths = [28]#,28
for depth in depths:
    batch_size = 32
    prediction_horizon = 1
    test_precent=0.1
    df_result = pd.DataFrame(columns=['model', 'experiment_id', 'mae', 'r'])
    df_idx=0
    for seed in range(0,1):
        random.seed(seed)
        dir=r'GCC_data/'
        files=os.listdir(dir)
        target = 'Gcc'
        cols = [
            'SW_IN_ERA',
            'TA_ERA',
            'TA_ERA_DAY',
            'TA_ERA_NIGHT'
                ]
        # ###############
        # dir = r'polution_data/'
        # files = os.listdir(dir)
        # target = 'pm2.5'
        # cols = [
        #     'DEWP',
        #     'TEMP',
        #     'PRES',
        #     'cbwd',
        #     'Iws',
        #     'Is',
        #     'Ir'
        # ]
        train_datasets={}
        val_datasets={}
        test_datasets={}

        start=True
        for file in files:
            print(file)
            file_path=dir+file
            data = pd.read_csv(file_path)
            site_colms=data.columns.values.tolist()
            site_final_cols=copy.deepcopy(cols)
            length=data.shape[0]
            data1=data
            X_train1 = np.zeros((len(data1), depth, len(site_final_cols)))
            for i, name in enumerate(site_final_cols):
                for j in range(depth):
                    X_train1[:, j, i] = data1[name].shift(depth - j - 1)#.fillna(method="bfill")
            y_train1 = np.array(data1[target].shift(-prediction_horizon))#.fillna(method='ffill'))
            X_train1 = X_train1[depth:-prediction_horizon]
            y_train1 = y_train1[depth:-prediction_horizon]
            y_nan_mask=np.isnan(y_train1) | np.isinf(y_train1)
            y_extreme_mask=np.abs(y_train1)>pow(10,5)
            y_nan_mask=(y_nan_mask | y_extreme_mask)
            x_temp=X_train1.reshape((X_train1.shape[0],-1)).copy()
            x_nan_mask=np.any(np.isnan(x_temp),axis=1) | np.any(np.isinf(x_temp),axis=1)
            x_extreme_mask=np.any(np.abs(x_temp)>pow(10,5),axis=1)
            x_nan_mask=(x_nan_mask|x_extreme_mask)
            nan_mask=(y_nan_mask | x_nan_mask)
            if np.sum(nan_mask==False)>0:
                X_train1=X_train1[nan_mask==False]
                y_train1 = y_train1[nan_mask == False]
                all_idxs = [item_idx for item_idx in range(y_train1.shape[0])]
                test_idxs = random.sample(all_idxs, int(test_precent * len(all_idxs)))
                train_validate_idxs = list(set(all_idxs).difference(set(test_idxs)))
                validate_idxs = random.sample(train_validate_idxs, int(1 / 9 * len(train_validate_idxs)))
                train_idxs = np.array(list(set(train_validate_idxs).difference(set(validate_idxs))))
                test_idxs = np.array(test_idxs)
                validate_idxs = np.array(validate_idxs)
                if start:
                    train_datasets['X']=X_train1[train_idxs]
                    train_datasets['Y'] = y_train1[train_idxs]
                    test_datasets['X'] = X_train1[test_idxs]
                    test_datasets['Y'] = y_train1[test_idxs]
                    val_datasets['X'] = X_train1[validate_idxs]
                    val_datasets['Y'] = y_train1[validate_idxs]
                    start=False
                else:
                    train_datasets['X'] = np.concatenate([train_datasets['X'],X_train1[train_idxs]], axis=0)
                    train_datasets['Y'] = np.concatenate([train_datasets['Y'],y_train1[train_idxs]], axis=0)
                    test_datasets['X'] = np.concatenate([test_datasets['X'], X_train1[test_idxs]],
                                                                    axis=0)
                    test_datasets['Y'] = np.concatenate([test_datasets['Y'], y_train1[test_idxs]],axis=0)
                    val_datasets['X'] = np.concatenate(
                        [val_datasets['X'], X_train1[validate_idxs]], axis=0)
                    val_datasets['Y'] = np.concatenate(
                        [val_datasets['Y'], y_train1[validate_idxs]], axis=0)
        print(train_datasets.keys())

        x_train=train_datasets['X']
        y_train=train_datasets['Y']
        x_test = test_datasets['X']
        y_test = test_datasets['Y']
        x_validate = val_datasets['X']
        y_validate = val_datasets['Y']
        x_all = np.concatenate((x_train, x_validate, x_test), axis=0)
        y_all = np.concatenate((y_train, y_validate, y_test), axis=0)
        x_means=np.nanmean(x_all.reshape((-1,x_all.shape[2])),axis=0)
        x_std=np.nanstd(x_all.reshape((-1,x_all.shape[2])),axis=0)
        y_means=np.nanmean(y_all,axis=0)
        y_std = np.nanstd(y_all, axis=0)
        x_train=(x_train-x_means)/(x_std+pow(10,-6))
        y_train=(y_train-y_means)/(y_std+pow(10,-6))
        x_test=(x_test-x_means)/(x_std+pow(10,-6))
        y_test=(y_test-y_means)/(y_std+pow(10,-6))
        x_validate = (x_validate - x_means) / (x_std + pow(10, -6))
        y_validate = (y_validate - y_means) / (y_std + pow(10, -6))

        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        x_validate = torch.Tensor(x_validate)
        y_validate = torch.Tensor(y_validate)

        train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_validate, y_validate), shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)

        device='cpu'
        him_dim=4
        model = LSTMNet(x_train.shape[2],  him_dim, 1,device=device,dropout_rate=0.1).to(device=device)#.cuda()#input_dim, hidden_dim, output_dim,dropout_rate=0,device='cpu'
        opt = torch.optim.Adam(model.parameters(), lr=0.01,)#weight_decay=pow(10,-3)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        epochs = 40
        loss = nn.MSELoss()
        patience = 50
        min_val_loss = 9999
        counter = 0
        para_path=f"lstm_phenology_{him_dim}_seed{seed}_lag{depth}.pt"
        if os.path.exists(para_path):
            model.load_state_dict(torch.load(para_path))
        for i in range(epochs):
            mse_train = 0
            iteration_start = time.monotonic()
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device=device)#.cuda()
                batch_y = batch_y.to(device=device)#.cuda()
                opt.zero_grad()
                y_pred= model(batch_x)
                y_pred = y_pred.squeeze(1)
                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item() * batch_x.shape[0]
                opt.step()
            epoch_scheduler.step()
            # validate
            with torch.no_grad():
                model.eval()
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device=device)#.cuda()
                    batch_y = batch_y.to(device=device)#.cuda()
                    output= model(batch_x)
                    output = output.squeeze(1)
                    preds.append(output.cpu().numpy())
                    true.append(batch_y.cpu().numpy())
                    mse_val += loss(output, batch_y).item() * batch_x.shape[0]
            preds = np.concatenate(preds)
            true = np.concatenate(true)

            if min_val_loss > mse_val ** 0.5:
                min_val_loss = mse_val ** 0.5
                print("Saving...")
                torch.save(model.state_dict(), para_path)
                counter = 0
            else:
                counter += 1

            if counter == patience:
                break
            print("Iter: ", i, "train: ", (mse_train / len(x_train)) ** 0.5, "val: ", (mse_val / len(x_train)) ** 0.5)
            iteration_end = time.monotonic()
            print("Iter time: ", iteration_end - iteration_start)
            if (i % 10 == 0):
                preds = preds * (y_std+pow(10,-6)) + y_means
                true = true * (y_std+pow(10,-6)) + y_means
                mse = mean_squared_error(true, preds)
                mae = mean_absolute_error(true, preds)
                r=stats.pearsonr(true, preds)[0]
                print("mse: ", mse, "mae: ", mae,'r:',r)
                # plt.figure(figsize=(6, 5))
                # plt.plot(preds)
                # plt.plot(true)
                # plt.show()

        # test
        model.load_state_dict(torch.load(para_path))
        with torch.no_grad():
            model.eval()
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device=device)#.cuda()
                batch_y = batch_y.to(device=device)#.cuda()
                output= model(batch_x)
                output = output.squeeze(1)
                preds.append(output.cpu().numpy())
                true.append(batch_y.cpu().numpy())
                mse_val += loss(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        preds = preds*(y_std+pow(10,-6)) + y_means
        true = true*(y_std+pow(10,-6)) + y_means

        mse = mean_squared_error(true, preds)
        mae = mean_absolute_error(true, preds)
        plt.scatter(true,preds)
        plt.show()
        r=stats.pearsonr(true, preds)[0]
        temp_result = ['lstm', seed, mae, r]
        df_result.loc[df_idx] = temp_result
        df_idx += 1
        print(type, mse, mae,r)
    df_result.to_csv(f'lstm_mae_r_lag{depth}.csv')
