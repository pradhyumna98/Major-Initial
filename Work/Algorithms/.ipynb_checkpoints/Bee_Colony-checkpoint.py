import random,timeit
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as ms

def random_search(n,dim):
    gens=[[0 if g != j else 1 for g in range(n)] for j in range(dim)]
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

def bees_optimization(bee,binary,i):
    binary=list(binary)
    j=random.randint(0,len(binary)-1)
    k=random.randint(0,len(binary)-1)
    while k==i:
        k=random.randint(0,len(binary)-1)
    fit=binary[j]+random.uniform(-1,1)*(binary[j]-binary[k])
    for x in range(bee):
        y=random.randint(0,len(binary)-1)
        while y==i:
            y=random.randint(0,len(binary)-1)
        r=random.uniform(0,1)
        if r<=fit:
            binary[y]=1
        
    return binary

def BABCO(train_d,train_l,n=10,max_iter=25,employee_percent=0.5,max_limit=5):
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
    employed_bees = int(round(n*employee_percent))
    onlooker_bees = n - employed_bees       

    dim=len(train_d[0])
    global_best=float("-inf")
    global_position=tuple([0]*dim)
    gens_dict = {}
    limit=[0]*dim
    gens=random_search(dim,dim)
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
        for i in range(employed_bees):
            for i,x in enumerate(gens):
                gen=bees_optimization(employed_bees,x,i)
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=evaluate(train_d,train_l,gen)
                    gens_dict[tuple(gen)]=score

                if score > gens_dict[tuple(gens[i])]:
                    limit[i]=0
                    gens[i]= gen
                else:
                    limit[i]+=1

                if score > global_best:
                    global_best=score
                    global_position=dc(gen)

                if limit[i]>=max_limit:
                    gens[i]=[0 if g != i else 1 for g in range(dim)]
    
        for i in range(onlooker_bees):
            for i,x in enumerate(gens):
                gen=bees_optimization(employed_bees,x,i)
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=evaluate(train_d,train_l,gen)
                    gens_dict[tuple(gen)]=score

                if score > gens_dict[tuple(gens[i])]:
                    limit[i]=0
                    gens[i]= gen
                else:
                    limit[i]+=1

                if score > global_best:
                    global_best=score
                    global_position=dc(gen)

                if limit[i]>=max_limit:
                    gens[i]=[0 if g != i else 1 for g in range(dim)]

                
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

def Bee_Colony(k,train_d,test_d,train_l,test_l):
#     k=[1 for r in range(len(x[0]))]
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
        g,l=BABCO(train_d,train_l,n=10,max_iter=25,employee_percent=0.5,max_limit=5)
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