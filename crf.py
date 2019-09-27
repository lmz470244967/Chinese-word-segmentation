# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:34 2019

@author: lmz470244967
"""
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
 
'''
  Args:
    predicted:  unary-probability, [batch-size, max-sequence-len, tag-num]
    gold:       referred tag index, [batch-size,max-sequence-len]
    transitions:transition probability from tag i to tag j, [tag-num, tag-num]
    length:     the true length of each sentence, [batch-size]

  Returns:
    score: a computed crf positive number in pytorch tensor
'''


def to_scalar(var): #var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]
 
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

#unequaled length input *************************
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def crf_log_likelihood(predicted,gold,num_tag,transitions,length):
    # predicted including each num-tag, so, the shape is [batch-size,sentence-length,num-tag]
    # gold only give the tag index, so, the shape is [batch-size,sentence-length]
    forward_score = torch.zeros(1)
    realpat_score = torch.zeros(1)
    for pred, go, leng in zip(predicted,gold,length):
        pred = pred[:leng]
        go = go[:leng]
        forward_score += forward_alg(pred,num_tag,transitions)
        realpat_score += gold_alg(pred,go,num_tag,transitions)
    
    print(forward_score)
    print(realpat_score)
    return forward_score - realpat_score
    
    
def forward_alg(predicted,num_tag,transitions):
    # shape [1,4]
    init_alphas = torch.full((1, num_tag), -10000.)
    transitions = transitions.transpose(1,0)
    #***********************************adapt for CWS task***********************************
    #S, B can be as start tag
    init_alphas[0][0] = 0
    init_alphas[0][3] = 0
    # word -> stop
    end = torch.Tensor([0.,-10000.,-10000.,0.])
    
    forward_var = init_alphas
    for feat in predicted:
        #t step forward tensors
        alphas_t = []
        for next_tag in range(num_tag):
            #[0.1,0.2,0.3,0.4]  =>  [[0.1,0.1,0.1,0.1]] for the first cycle
            emit_score = feat[next_tag].view(1,-1).expand(1, num_tag)
            trans_score = transitions[next_tag].view(1,-1)
            next_tag_var = forward_var + trans_score + emit_score
            # scores
            alphas_t.append(log_sum_exp(next_tag_var).view(1))
        
        forward_var = torch.cat(alphas_t).view(1,-1)
    terminal_var = forward_var + end
    alpha = log_sum_exp(terminal_var)
    return alpha

def gold_alg(predicted,gold,num_tag,transitions):
    score = torch.zeros(1)
    gold = torch.cat([torch.IntTensor([0]), gold],0)
    transitions = transitions.transpose(1,0)
    end = torch.Tensor([0.,-10000.,-10000.,0.])
    
    for i, feat in enumerate(predicted):
        score = score + transitions[gold[i + 1], gold[i]] + feat[gold[i + 1]]
    score = score + end[gold[-1]]
    return score



    