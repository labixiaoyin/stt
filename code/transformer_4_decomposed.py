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
    def __init__(self,d_model, dk, n_heads,dff):
        super(EncoderLayer, self).__init__()

        
        self.attn1 = MultiHeadAttentionBlock(d_model, dk , n_heads)
        self.attn2 = MultiHeadAttentionBlock(d_model, dk , n_heads)
        self.fc1 = nn.Linear(dff, d_model)
        self.fc2 = nn.Linear(d_model, dff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        B,T,n,d_model=x.size()
        
        a=torch.transpose(x,1,2).contiguous().view(-1,T,d_model)
        a = self.attn1(a)
        a=a.view(-1,n,T,d_model)
        a=torch.transpose(a,1,2)
        x = self.norm1(x + a)
        
        a=x.view(-1,n,d_model)
        a=self.attn2(a)
        a=a.view(-1,T,n,d_model)
        x=self.norm2(x+a)
        
        a = self.fc1(F.elu(self.fc2(x)))      
        x = self.norm3(x + a)
        
        return x
    
class QGB(torch.nn.Module):
    def __init__(self,para,n):
        super(QGB, self).__init__()
        self.n=n
        self.para=para
        
        self.LN1=nn.Linear(1,self.para.d_model)
        self.LN2=nn.Linear(self.n*self.para.window,self.para.horizon)
        self.LN3=nn.Linear(self.para.window*n,self.para.horizon)
        self.pos = PositionalEncoding(self.para.d_model)
            
    def forward(self,x,E_x):
        #x:[batch,T,n]
        Hx=x.view(-1,self.para.window*self.n,1)
        Hx=self.LN1(Hx)      #Hx:[batch,Tn,d_model]
        Hx=F.elu(Hx)
        Hx=torch.transpose(Hx,1,2) #Hx:[bathc,d_model,Tn]
        Hx=self.LN2(Hx)  #Hx:[bathc,d_model,horzion]
        Hx=torch.transpose(Hx,1,2)
        
        #E_x:[batch,Tn,d_model]
        Hz=torch.transpose(E_x,1,2)
        Hz=self.LN3(Hz)
        Hz=torch.transpose(Hz,1,2)   #E_x:[batch,horizon,d_model]
        
        Zd=Hx+Hz
        Zd=self.pos(Zd)

        return Zd

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
        x = self.norm1(a + x)
       
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a)

        return x

class Transformer(torch.nn.Module):
    def __init__(self,para, drop_prob, n):
        super(Transformer, self).__init__()
        self.para=para
        self.batch=para.batch_size
        self.n=n
        self.T=para.window
        self.dim_val=para.dim_val
        self.d_k=para.d_k
        
        #Initiate encoder and Decoder layers
        
        #Encoder
        self.encs = nn.ModuleList()
        for i in range( para.n_layers):
             self.encs.append(EncoderLayer(self.para.d_model,self.d_k, para.n_heads,para.dff))
        
        #QGB
        self.QGB=QGB(para,n)
                
        #Decoder
        self.decs = nn.ModuleList()
        for i in range(para.n_layers):
            self.decs.append(DecoderLayer(self.para.d_model, self.d_k, para.n_heads,para.dff))


        #Dense layers for managing network inputs and outputs
        self.pos = PositionalEncoding(self.n)
        self.enc_input_fc = nn.Linear(1, self.para.d_model)
        self.out_fc = nn.Linear(self.para.horizon*self.para.d_model,self.para.horizon)
    
    def forward(self, x):
        E=self.pos(x)
        E=E.view(-1,self.T,self.n,1)
        E=self.enc_input_fc(E)   #E:[batch,T,n,d_model]

        #encoder_S
        e=self.encs[0](E)
        for enc in self.encs[1:]:
            e = enc(e)   #e:[batch,T,n,d_model]
            
        e=e.view(-1,self.T*self.n,self.para.d_model)
        Zd=self.QGB(x,e)

        #decoder
        d = self.decs[0](Zd, e)
        for dec in self.decs[1:]:
            d = dec(d, e)
        
        #output
        x = self.out_fc(d.flatten(start_dim=1))
        x=F.elu(x)
        return x
