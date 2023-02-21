# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:38:38 2021

@author: YCR
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, dk, n_heads,dff):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttentionBlock(d_model, dk , n_heads)
        self.fc1 = nn.Linear(dff, d_model)
        self.fc2 = nn.Linear(d_model, dff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        a = self.attn(x)
        a=self.dropout(a)
        
        x = self.norm1(x + a)
        x=self.dropout(x)
        
        a = self.fc1(F.elu(self.fc2(x)))
        a=self.dropout(a)
        
        x = self.norm2(x + a)
        x=self.dropout(x)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, dk, n_heads,dff):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(d_model, dk, n_heads)
        self.attn2 = MultiHeadAttentionBlock(d_model, dk, n_heads)
        self.fc1 = nn.Linear(dff, d_model)
        self.fc2 = nn.Linear(d_model, dff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x, enc):
        a = self.attn1(x)
        a=self.dropout(a)
        
        x = self.norm1(a + x)
       
        x=self.dropout(x)

        a = self.attn2(x, kv = enc)
        a=self.dropout(a)
        
        x = self.norm2(a + x)
        x=self.dropout(x)

        a = self.fc1(F.elu(self.fc2(x)))
        a=self.dropout(a)
        x = self.norm3(x + a)
        x=self.dropout(x)

        return x
    
class QGB(torch.nn.Module):
    def __init__(self,para,n):
        super(QGB, self).__init__()
        self.n=n
        self.para=para
        
        self.LN1=nn.Linear(1,self.para.d_model)
        self.LN2=nn.Linear(self.n*self.para.window,self.para.horizon)
        self.LN3=nn.Linear(self.para.window,self.para.horizon)
        self.pos = PositionalEncoding(self.para.d_model)
        
        self.dropout = nn.Dropout(p=0.1)
            
    def forward(self,x,E_x):
        #x:[batch,T,n]
        Hx=x.view(-1,self.n*self.para.window,1)
        Hx=self.LN1(Hx)      #Hx:[batch,Tn,d_model]
        Hx=self.dropout(F.elu(Hx))
        
        Hx=torch.transpose(Hx,1,2) #Hx:[bathc,d_model,Tn]
        Hx=self.dropout(self.LN2(Hx))  #Hx:[bathc,d_model,horzion]
        Hx=torch.transpose(Hx,1,2)
        
        #E_x:[batch,T,d_model]
        Hz=torch.transpose(E_x,1,2)
        Hz=self.dropout(self.LN3(Hz))
        Hz=torch.transpose(Hz,1,2)   #E_x:[batch,horizon,d_model]
        
        Zd=Hx+Hz
        Zd=self.pos(Zd)

        return Zd
        

class Transformer(torch.nn.Module):
    def __init__(self, para,drop_prob, input_size):
        super(Transformer, self).__init__()
        self.para = para
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.dim_val=para.dim_val
        self.i_len=para.window
        self.input_size=input_size
        self.dim_attn=para.d_k
        self.d_model=para.d_model
        
        
        #Initiate encoder and Decoder layers
        
        self.QGB=QGB(para,input_size)
        
        #encoder_temporal
        self.encs_t = nn.ModuleList()
        for i in range(para.n_layers):
            self.encs_t.append(EncoderLayer(self.d_model,self.dim_attn, para.n_heads,para.dff))
        
        #encoder_spital
        self.encs_s=nn.ModuleList()
        for i in range(para.n_layers):
            self.encs_s.append(EncoderLayer(self.d_model, self.dim_attn, para.n_heads,para.dff))
        
        self.decs = nn.ModuleList()
        for i in range(para.n_layers):
            self.decs.append(DecoderLayer(self.d_model, self.dim_attn, para.n_heads,para.dff))
        
        self.pos = PositionalEncoding(self.d_model)
        
        #Dense layers for managing network inputs and outputs
        self.LN1 = nn.Linear(self.input_size, self.d_model)
        self.LN2=nn.Linear(self.i_len,self.d_model)
        self.LN3=nn.Linear(self.i_len+self.input_size,self.i_len)

        self.out_fc = nn.Linear(self.para.horizon*self.d_model,self.para.horizon)
            
    def forward(self, x):
        t_x=self.LN1(x)  
        t_x=self.dropout(self.pos(t_x))   #_x [batch,T,d_model]
        
        #temporal
        e_t=self.encs_t[0](t_x)
        for enc_t in self.encs_t[1:]:
            e_t=enc_t(e_t)   #e_t:[batch,T,d_model]
               
            
        Zd=self.QGB(x,e)   #Zd:[batch,horizon,d_model]
        
        d = self.decs[0](Zd,e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        #output
        d = self.out_fc(d.flatten(start_dim=1))
        d=F.elu(d)
        
        return d
