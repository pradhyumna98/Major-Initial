import random,math,timeit
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as ms

def random_search(n,dim):
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens



def case1(move):
    return 1 if random.uniform(-0.1,0.9)<move else 0
def case2(one_bin):
    if random.uniform(-0.1,0.9)<math.tanh(int(one_bin)):
        if one_bin==1:
            return 0
        else:return 1
    else:return 0
def exchange_binary(binary,score,alpha,beta,gamma):
    al_binary=binary
    e=0.05*random.uniform(0,1)
    movement=beta*math.exp(-gamma*score**2)*score + alpha*e
    if random.uniform(0,1) < movement:
        for i,b in enumerate(binary):
            move=beta*math.exp(-gamma*b**2)*b + alpha*e
            al_binary[i]=case1(move)
    else:
        for i,b in enumerate(binary):
            move=beta*math.exp(-gamma*b**2)*b + alpha*e
            al_binary[i]=case2(move)
    return al_binary

def evaluate(train_d,train_l,gen):
        mask=np.array(gen) > 0
        al_data=np.array([al[mask] for al in train_d])
        kf = ms.KFold(n_splits=4)
        s = 0
        for tr_ix,te_ix in kf.split(al_data):
            s+= RandomForestClassifier(n_estimators=25).fit(al_data[tr_ix],train_l[tr_ix]).score(al_data[te_ix],train_l[te_ix])#.predict(al_test_data)
        s/=4
        return s

def BFA(train_d,train_l,n=20,max_iter=25,gamma=0.20,beta=0.20,alpha=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            max_iter: Number of max iteration, default=300
            }
    output:{
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    dim=len(train_d[0])
    global_best=float("-inf")
    global_position=tuple([0]*dim)
    gens_dict = {tuple([0]*dim):float("-inf")}
    gens=random_search(n,dim)
    for gen in gens:
        if tuple(gen) in gens_dict:
            score = gens_dict[tuple(gen)]
        else:
            score=evaluate(train_d,train_l,gen)
            gens_dict[tuple(gen)]=score
        if score > global_best:
            global_best=score
            global_position=dc(gen)
    for it in range(max_iter):
        for i,x in enumerate(gens):
            for j,y in enumerate(gens):
                if gens_dict[tuple(y)] < gens_dict[tuple(x)]:
                    gens[j]=exchange_binary(y,gens_dict[tuple(y)],alpha,beta,gamma)
                gen = gens[j]
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=evaluate(train_d,train_l,gens[j])
                    gens_dict[tuple(gen)]=score
                if score > global_best:
                    global_best=score
                    global_position=dc(gen)
    return global_position,global_position.count(1)


def test_score(gen,tr_x,tr_y,te_x,te_y):
    mask=np.array(gen) == 1
    al_data=np.array(tr_x[:,mask])
    al_test_data=np.array(te_x[:,mask])
    return np.mean([RandomForestClassifier(n_estimators=25).fit(al_data,tr_y).score(al_test_data,te_y) for i in range(5)])

def listToString(s):  
    str1 = ""   
    for ele in s:  
        str1 += str(ele)     
    return str1

def Bacterial_Foraging(k,train_d,test_d,train_l,test_l):
   # k=[1 for r in range(len(x[0]))]
    #print(test_score(k,train_d,train_l,test_d,test_l))
    max_1 = test_score(k,train_d,train_l,test_d,test_l)
    colo = listToString(k)
    print(max_1)
    fattr=0
    ftest=0.0
    flist=[0 for r in range(len(k))]
    final_list=[0 for r in range(len(k))]
    start=timeit.default_timer()
    for i in range(20):
        g,l=BFA(train_d,train_l,n=20,max_iter=25,gamma=0.20,beta=0.20,alpha=0.25)
        fattr+=l
        test=test_score(g,train_d,train_l,test_d,test_l)
        if test>max_1:
            max_1 = test
            colo= "".join(map(str,g))
        ftest+=test
        for j in range(len(flist)):
            if g[j]==1:
                flist[j]+=1
        print("{0}  {1}  {2}  {3:.6f}".format(i+1,"".join(map(str,g)),l,test))
    fattr=fattr//20
    ftest=ftest/20
    end=timeit.default_timer()
    time=end-start
    print(flist,fattr)
    final=np.argsort(flist)[::-1][:fattr]
    print(final)
    for i in range(len(final)):
        final_list[final[i]]=1
    print("{0}  {1}   {2}   {3:.6f}    {4:.4f}".format("Final: ","".join(map(str,final_list)),fattr,ftest,time))
    return max_1,colo