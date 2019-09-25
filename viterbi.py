# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:55:43 2019

@author: lmz470244967
new viterbi
"""
import numpy as np
import evaluate as ev
import time

transition = np.array([[0.5,0.5,-9999,-9999],
                  [-9999,-9999,0.5,0.5],
                  [-9999,-9999,0.5,0.5],
                  [0.5,0.5,-9999,-9999]])

#viterbi decode released by tensorflow
def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

#viterbi with start and end matrix & using multiply instead of plus 
def viterbi(y):
    start = np.array([0.5,0.5,0.,0.])
    transition = np.array([[0.5,0.5,0.,0.],
                  [0.,0.,0.5,0.5],
                  [0.,0.,0.5,0.5],
                  [0.5,0.5,0.,0.]])
    end = np.array([0.5,0.,0.,0.5])
    prob = np.array([1,1,1,1])
    route_box = []
    batch = []
    #start character init
    for sentence in y:
        sentence[0] = sentence[0] * start   #**************************
    
    #y shape [batch-size,sentence-length,tag-num], sentence shape [sentence-length,tag-num]
    for sentence in y:
        #reset
        prob = np.array([1,1,1,1])
        best = []
        route_box = []
        for word in sentence:
            #word shape [tag-num]
            word = word * prob  #current state word probability  ****************
            vit = [word[i] * transition[i] for i in range(4)]  #next state transition prob  ****************
            vit = np.array(vit)
            route = np.argmax(vit,0)  #return index
            prob = np.max(vit,0)  #return prob
            route_box.append(route)  #used for back-routing
        #the last word has different transition matrix
        route_box.pop()
        word = word * end   #*********************
        last_index = np.argmax(word)
        
        #save the best route
        best = [last_index]
        
        #back-routing: reverse to find the most likely squence
        route_box.reverse()
        for route in route_box:
            best.append(route[last_index])
            last_index = route[last_index]
        
        best.reverse()
        batch.append(best)
    return batch

#test three viterbi algorithms
def similarity(x,y):
    temp = (x == y)
    total = x.size
    num = 0
    for sentence in temp:
        for word in sentence:
            if word:
                num+=1
    return num / total
    
    

if __name__ == "__main__":
    
    #batch-size, sentence-length, tag-number
    y = np.random.rand(2,8,4)
    
    astart = time.time()
    a = np.array(viterbi(y))
    aend = time.time()
    
    b = np.array(ev.viterbi(y))
    bend = time.time()
    
    c = []
    for i in y:
        c1, c2 = viterbi_decode(i,transition)
        c.append(c1)
    c = np.array(c)
    cend = time.time()
    
    
    
    print(a)
    print(b)
    print(c)
    '''
    print(similarity(a,b))
    print(similarity(a,c))
    print(similarity(b,c))
    
    print(str(aend-astart))
    print(str(bend-aend))
    print(str(cend-bend))
    '''
    

    