import random,math,copy,timeit
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



def evaluate(train_d,train_l,gen):
        mask=np.array(gen) > 0
        al_data=np.array([al[mask] for al in train_d])
        kf = ms.KFold(n_splits=4)
        s = 0
        for tr_ix,te_ix in kf.split(al_data):
            s+= RandomForestClassifier(n_estimators=25).fit(al_data[tr_ix],train_l[tr_ix]).score(al_data[te_ix],train_l[te_ix])#.predict(al_test_data)
        s/=4
        return s

def logsig(n): return 1 / (1 + math.exp(-n))
def sign(x): return 1 if x > 0 else (-1 if x!=0 else 0)

def BPSO(train_d,train_l,n=20,max_iter=200,w1=0.5,c1=0.5,c2=0.5,vmax=4):
    """
    input:{ 
            Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            max_iter: Number of max iteration, default=300
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            w1: move rate, default=0.5
            c1,c2: It's are two fixed variables, default=1,1
            vmax: Limit search range of vmax, default=4
            }
    output:{
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    dim=len(train_d[0])
    personal_best=float("-inf")
    global_best=float("-inf")
    gens=random_search(n,dim)
    vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    one_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    zero_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    fit=[float("-inf") for i in range(n)]
    personal_best=dc(fit)
    xpersonal_best=dc(gens)
    global_best=max(fit)
    xglobal_best=gens[fit.index(max(fit))]
    gens_dict={tuple([0]*dim):float("-inf")}
    for it in range(max_iter):
        for i in range(n):
            if tuple(gens[i]) in gens_dict:
                score=gens_dict[tuple(gens[i])]
            else:
                score=evaluate(train_d,train_l,gens[i])
                gens_dict[tuple(gens[i])]=score
            fit[i]=score
            if fit[i]>personal_best[i]:#max
                personal_best[i]=dc(fit[i])
                xpersonal_best[i]=dc(gens[i])
        gg=max(fit)
        xgg=gens[fit.index(gg)]
        if global_best<gg:#max
            global_best=dc(gg)
            xglobal_best=dc(xgg)
        oneadd=[[0 for d in range(dim)] for i in range(n)]
        zeroadd=[[0 for d in range(dim)] for i in range(n)]
        c3=c1*random.random()
        dd3=c2*random.random()
        for i in range(n):
            for j in range(dim):
                if xpersonal_best[i][j]==0:
                    oneadd[i][j]=oneadd[i][j]-c3
                    zeroadd[i][j]=zeroadd[i][j]+c3
                else:
                    oneadd[i][j]=oneadd[i][j]+c3
                    zeroadd[i][j]=zeroadd[i][j]-c3
                if xglobal_best[j]==0:
                    oneadd[i][j]=oneadd[i][j]-dd3
                    zeroadd[i][j]=zeroadd[i][j]+dd3
                else:
                    oneadd[i][j]=oneadd[i][j]+dd3
                    zeroadd[i][j]=zeroadd[i][j]-dd3
        one_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(one_vel,oneadd)]
        zero_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(zero_vel,zeroadd)]
        for i in range(n):
            for j in range(dim):
                if abs(vel[i][j]) > vmax:
                    zero_vel[i][j]=vmax*sign(zero_vel[i][j])
                    one_vel[i][j]=vmax*sign(one_vel[i][j])
        for i in range(n):
            for j in range(dim):
                if gens[i][j]==1:
                    vel[i][j]=zero_vel[i][j]
                else:
                    vel[i][j]=one_vel[i][j]
        veln=[[logsig(s[_s]) for _s in range(len(s))] for s in vel]
        temp=[[random.random() for d in range(dim)] for _n in range(n)]
        for i in range(n):
            for j in range(dim):
                if temp[i][j]<veln[i][j]:
                    gens[i][j]= 0 if gens[i][j] ==1 else 1
                else:
                    pass
    return xglobal_best,xglobal_best.count(1)

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

def Particle_Swarm(k,train_d,test_d,train_l,test_l):
    #k=[1 for r in range(len(x[0]))]
    max_1 = test_score(k,train_d,train_l,test_d,test_l)
    colo = listToString(k)
    print(max_1)
    fattr=0
    ftest=0.0
    flist=[0 for r in range(len(k))]
    final_list=[0 for r in range(len(k))]
    start=timeit.default_timer()
    for i in range(20):
        g,l=BPSO(train_d,train_l,n=20,max_iter=200,w1=0.5,c1=0.5,c2=0.5,vmax=4)
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