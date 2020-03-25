import random,math,copy,timeit,operator
from sklearn import utils,preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy as dc
from sys import exit
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as ms

class ants(object):
    def __init__(self,gens=None,mse=None):
        self.gens=gens
        self.mse=mse

def random_search(n,dim):
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens



def k_search(n,dim,flist,pheromone):
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
        for j,x in enumerate(gen):
            gen[j]=flist[j]*gen[j]
    return gens


def evaluate(train_d,train_l,gen):
        mask=np.array(gen) > 0
        al_data=np.array([al[mask] for al in train_d])
        kf = ms.KFold(n_splits=4)
        s = 0
        for tr_ix,te_ix in kf.split(al_data):
            s+= RandomForestClassifier(n_estimators=25).fit(al_data[tr_ix],train_l[tr_ix]).score(al_data[te_ix],train_l[te_ix])#.predict(al_test_data)
        s/=4
        return s

def BACO(train_d,train_l,n=30,max_iter=25,T0=0.1,k=10,m=25,p=3,rho=0.25):
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
    pheromone=list([T0]*dim)
    pher_diff=list([0]*dim)
    global_position=tuple([0]*dim)
    antslist=[]
    gens_dict = {}
    for i in range(n):
        gens=random_search(m,dim)
        mse=0
        for gen in gens:
            if tuple(gen) in gens_dict:
                score = gens_dict[tuple(gen)]
            else:
                score=evaluate(train_d,train_l,gen)
                gens_dict[tuple(gen)]=score
            global_best=score
            global_position=dc(gen)
            mse+=score
        mse=mse/m
        antslist.append(ants(gens,mse))
    antslist.sort(key=operator.attrgetter('mse'), reverse=True)
    klist=list(antslist[i] for i in range(k))
    mselist=list(antslist[i].mse for i in range(k))
    for j in range(len(klist)):
        diff=(max(mselist)-mselist[j])/(max(max(mselist)-mselist))
        flist=list([0]*dim)
        for gen in klist[j].gens:
            for a,x in enumerate(gen):
                if x==1:
                    pher_diff[a]=diff
                    flist[a]=1
                else:
                    pher_diff[a]=0
    
    for i in range(len(pheromone)):
        pheromone[i]=pheromone[i]*rho + pher_diff[i]
        
        
    for it in range(max_iter):
        m=m-p
        for i in range(n):
            gens=k_search(m,dim,flist,pheromone)
            mse=0
            for gen in gens:
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=evaluate(train_d,train_l,gen)
                    gens_dict[tuple(gen)]=score
                global_best=score
                global_position=dc(gen)
                mse+=score
            mse=mse/m
            antslist.append(ants(gens,mse))
        antslist.sort(key=operator.attrgetter('mse'), reverse=True)
        klist=list(antslist[i] for i in range(k))
        mselist=list(antslist[i].mse for i in range(k))
        for j in range(len(klist)):
            diff=(max(mselist)-mselist[j])/(max(max(mselist)-mselist))
            flist=list([0]*dim)
            for gen in klist[j].gens:
                for i,x in enumerate(gen):
                    if x==1:
                        pher_diff[i]=diff
                        flist[i]=1
                    else:
                        pher_diff[i]=0
    
        for i in range(len(pheromone)):
            pheromone[i]=pheromone[i]*rho + pher_diff[i]

        
    return global_position,global_position.count(1)


def test_score(gen,tr_x,tr_y,te_x,te_y):
    mask=np.array(gen) == 1
    al_data = np.array(tr_x[:,mask])
    al_test_data = np.array(te_x[:,mask])
    return np.mean([RandomForestClassifier(n_estimators=25).fit(al_data,tr_y).score(al_test_data,te_y) for i in range(5)])

def listToString(s):  
    str1 = ""   
    for ele in s:  
        str1 += str(ele)     
    return str1

def Ant_Colony(k,train_d,test_d,train_l,test_l):
   
    #k=[1 for r in range(len(x[0]))]
    max_1 = test_score(k,train_d,train_l,test_d,test_l)
    colo = listToString(k)
    print(colo)
    print(max_1)
    fattr=0
    ftest=0.0
    flist=[0 for r in range(len(k))]
    final_list=[0 for r in range(len(k))]
    start=timeit.default_timer()
    for i in range(20):
        g,l=BACO(train_d,train_l,n=30,max_iter=25,T0=0.1,k=10,m=25,p=3,rho=0.25)
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