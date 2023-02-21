# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:01:49 2021

@author: YCR
"""

from get_data  import *
from Params_setup import *

#from transformer_1 import *
from transformer_2_parllel import *
#from transformer_3_joint import *
#from transformer_4_decomposed import *
from AR_Model import Ar

from LSTM_Net import Lstm
from Seq2seq_Net import Seq2seq
from TPA import Tpa
import torch
import torch.nn as nn
import Optim
import math
import time
import csv


def save_csv(data, X, Y, model,batch_size,net_op,):
    model.eval();
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
    
    print(predict)
    print(test)
    #print(predict.shape)
    
    predict=predict.detach().numpy().tolist()
    path='./predict_data/'+net_op+'.csv'
    f=open(path,'w',newline='')
    writer=csv.writer(f)
    for i in predict:
        writer.writerow(i)
    f.close
            
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad();
        output = model(X);
        #scale = data.scale.expand(output.size(0), data.raw_columns)
        scale = data.scale[-1].expand(output.size(0), output.size(1))
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * data.horizon);
    return total_loss / n_samples

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = data.scale[-1].expand(output.size(0), data.horizon)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.horizon);
    
    #评价指标
    rmse= math.sqrt(total_loss / n_samples)
    mae = total_loss_l1/n_samples
    
    #rse = math.sqrt(total_loss / n_samples)/data.rse
    #rae = (total_loss_l1/n_samples)/data.rae
    
    #correlation的计算
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    #return rse, rae, correlation;
    return rmse, mae, correlation;
    


# Set the random seed manually for reproducibility.
#torch.manual_seed(54321)

para=params_setup()

#data processing
Data = Data_utility(para)
print(Data.rse)

#create model
model = Transformer(para,para.drop_prob, Data.raw_columns)
#model =Lstm(para,Data)
#model=Tpa(para,Data)
#model=Ar(para,Data)
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

#determine the loss function
#这里sum的含义是为进行除以个数的操作
if para.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')

#determine the optimizer
best_val = 10000000;
optim = Optim.Optim(model.parameters(), para.optim, para.lr, para.clip,)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    for epoch in range(1, para.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, para.batch_size)
        val_rmse, val_mae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, para.batch_size);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rmse {:5.4f} | valid mae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse, val_mae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_rmse < best_val:
            with open(para.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_rmse
        if epoch % 5 == 0:
            test_rmse, test_mae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, para.batch_size);
            print ("test rmse {:5.4f} | test mae {:5.4f} | test corr {:5.4f}".format(test_rmse, test_mae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(para.save, 'rb') as f:
    model = torch.load(f)
test_rmse, test_mae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, para.batch_size);
print ("test rmse {:5.4f} | test mae {:5.4f} | test corr {:5.4f}".format(test_rmse, test_mae, test_corr))

#save_csv
save_csv(Data, Data.test[0], Data.test[1], model,para.batch_size,'new_h1_our')




