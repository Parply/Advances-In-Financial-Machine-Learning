import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import log_loss,accuracy_score
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import _BaseKFold
from scipy.stats import norm
from itertools import product
from random import gauss
import scipy.cluster.hierarchy as sch
import time
import multiprocessing as mp
import datetime as dt
import sys
import copy

def pcaWeights(cov,riskDist=None,riskTarget=1.):
    """
    PCA weights from a risk distribution R
    """
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov) # must be Hermitian (complete complex square matrix A with A[i,j] = complex conjugate of A[j,i])
    indicies = eVal.argsort()[::-1] # arguments for sorting
    eVal, eVec = eVal[indicies], eVec[:,indicies] # sort
    if riskDist is None: # make null dist
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1
    loads = riskTarget * (riskDist/eVal)**.5
    wghts = np.dot(eVec, np.reshape(loads,(-1,1)))
    return wghts


def getRolledSeries(pathIn,key):
    series = pd.read_hdf(pathIn,key=key)
    series['Time'] = pd.to_datetime(series['Time'],format='%Y%m%d%H%M%S%f')
    series = series.set_index('Time')
    gaps = rollGaps(series)
    for fld in ['Close','VWAP']:series[fld] -= gaps
    return series

def rollGaps(series,diction={'Instrument':'FUT_CUR_GEN_TICKER','Open':'PX_OPEN','Close':'PX_LAST'},matchEnd=True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[diction['Instrument']].drop_duplicates(keep='first').index
    gaps = series[diction['Close']]*0
    iloc = list(series.index)
    iloc = [iloc.index(i)-1 for i in rollDates] # index if dates prior to roll
    gaps.loc[rollDates[1:]] = series[diction['Open']].loc[rollDates[1:]] - series[diction['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd:gaps -= gaps.iloc[-1] # roll backward
    return gaps
    

def getDailyVol(close,span0=100):
    """
    Daily vol, reindexed to close
    """
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

def getTEvents(gRaw,h):
    """
    symmetric CUSUM filter
    """
    tEvents,sPos,sNeg = [],0,0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg = max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        if sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def applyPtSlOnTl(close,events,ptSl,molecule):
    """
    Triple-barrier labelling method

    close = pandas series of prices
    events = pandas dataframe with columns:
        t1 = timestamp of vertical barrier. When the value is np.nan, there will be no vertical barrier
        trgt = the unit width of the horizontal barriers
    ptSl = a list of two non-negative float values:
        ptSl[0] = The factor that multiplies the trgt to set the width of the upper barrier. If 0 no barrier
        ptSl[1] = The factor that multiplies the trgt to set the width of the lower barrier. If 0 no barrier
    molecule = a list with the subset of event indicies that will be processed by a single thread 
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0: pt=ptSl[0]*events_['trgt']
    else: pt=pd.Series(index=events.index) # NaNs
    if ptSl[1] > 0: sl=ptSl[1]*events_['trgt']
    else:sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # path prices
        df0 = (df0/close[loc] - 1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl'] = df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt'] = df0[df0>pt[loc]].index.min() # earliest profit taking
    return out

def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False,side=None):
    """
    Getting time of first touch of barriers

    close = pandas series of prices
    tEvents = pandas timeindex containing the timestamps that will seed every triple barrier
    ptSl = A non negative float that sets the width of the two barriers
    t1 = a pandas series with the timestamps of the verical barriers.
    trgt = a pandas series of targets, expressed as absolute returns
    minRet = the min target return required for running the triple barrier search
    numThreads = number of threads used by the function concurrently
    """
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # min retention 
    # 2) get t1 max holding period
    if t1 is False: t1 = pd.Series(pd.NaT,index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None: side_,ptSl_ = pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: side_,ptSl_ = side.loc[trgt.index],ptSl[:2]
    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnTl,pdObj=('molecule',events.index),numThreads=numThreads,close=close,events=events,ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None: events = events.drop('side',axis=1)
    return events

def getBins(events,close):
    """
    Labelling for side and size

    events.index = event's starttime
    events['t1'] = event's endtime
    events['trgt'] = event's target
    events['side'] = implies algo's position side (optional)
        Case 1: ('side' not in events): bin in (-1,1) <- label by price action
        Case 2: ('side' in events): bin in (0,1) <- label by pnl (meta-labelling)

    output dataframe columns:
    ret = the return realized at the time of the first touched barrier
    bin = label [-1,0,1] as a function of sign of an outcome
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px,method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index] - 1
    if 'side' in events_:out['ret'] *= events_['side'] # meta-labelling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labelling    
    out.loc[out.index==events['t1'],'bin'] = 0
    return out

def dropLabels(events,minPtc=.05):
    """
    drop under populated labels
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min()>minPtc or df0.shape[0]<3:break
        print('dropped label',df0.argmin(),df0.min()) 
        events = events[events['bin'] != df0.argmin()]
    return events

# Sample weights

def mpNumCoEvents(closeIdx,t1,molecule):
    """
    Compute number of concurrent events per bar
    molecule[0] = date of the first event on which the weight will be computed
    molecule[1] = date of the last event of which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count
    """
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1]) # unclosed events must still impact other events
    t1 = t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] # event that starts at or before t1[molecule].max()
    # 2) events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count = pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
    """
    average uniqueness of an event over its lifespan
    """
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

def getIndMatrix(barIx,t1):
    """
    Get indicator matrix
    """
    indM = pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i, (t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i] = 1.
    return indM

def getAvgUniqueness(indM):
    """
    Average uniqueness from indicator matrix
    """
    c = indM.sum(axis=1) # concurrency
    u = indM.div(c,axis=0) # uniqueness
    avgU = u[u>0].mean() # average uniqueness
    return avgU

def seqBootstrap(indM, sLength=None):
    """
    Generates a sample via sequential bootstrap
    """
    if sLength is None: sLength=indM.shape[1]
    phi = []
    while len(phi)<sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]] # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum() # get prob
        phi += [np.random.choice(indM.columns,p=prob)]
    return phi

def getRndT1(numObs,numBars,maxH):
    """
    random t1 series
    """
    t1 = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0,numBars)
        val = ix + np.random.randint(1,maxH)
        t1.loc[ix] = val
    return t1.sort_index()

def auxMC(numObs,numBars,maxH):
    """
    Uniqueness from standard and sequential bootstaps 

    Parallelised auxiliary function
    """
    t1 = getRndT1(numObs,numBars,maxH)
    barIx = range(t1.max()+1)
    indM = getIndMatrix(barIx,t1)
    phi = np.random.choice(indM.columns,size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU':stdU,'seqU':seqU}
    
def mainMC(numObs=10,numBars=100,maxH=5,numIters=1E6,numThreads=4):
    """
    multi-threaded monte carlo experiments
    """
    jobs=[]
    for i in range(int(numIters)):
        job = {'func':auxMC,'numObs':numObs,'numBars':numBars,'maxH':maxH}
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out = processJobs(jobs,numThreads=numThreads)
    print(pd.DataFrame(out).describe())
    return

def mpSampleW(t1,numCoEvents,close,molecule):
    """
    derive sample weight by return attribution
    """
    ret = np.log(close).diff() # log returns for additivity
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

def getTimeDecay(tW,clfLastW=1.):
    """
    apply piecewise linear decay to observed uniqueness (tW)
    newest observation gets weight=1, oldest get weight=clfLastW
    """
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0: slope = (1.-clfLastW)/clfW.iloc[-1]
    else: slope = 1./((clfLastW+1)*clfW.iloc[-1])
    const = 1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print(const,slope)
    return clfW

def getWeights(d,size):
    w = [1.]
    for k in range(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1,1)
    return w

def plotWeights(dRange,nPlots,size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_ = getWeights(d,size=size)
        w_ = pd.DataFrame(w_,index=range(w_.shape[0])[::1],columns=d)
        w = w.join(w_,how='outer')
    ax = w.plot()
    ax.legend(loc='upper left'); mpl.show()
    return

def fracDiff(series,d,thres=.01):
    """
    Increasing width window with treatment of NaNs
    Note 1: For thres=1 nothing is skipped
    Note 2: d can be any positive fractional not necessarily in [0,1]
    """
    # 1) compute weights for the longest series
    w = getWeights(d,series.shape[0])
    # 2) determine initial calcs skipped based on weight loss threshold
    w_ = np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    # 3) apply weights to values
    df = {}
    for name in series.columns:
        seriesF,df_ = series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc] = np.dot(w[-(iloc+1):,:],seriesF.loc[:loc])[0,0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df,axis=1)
    return df

# fixed window width

def getWeights_FFD(d,thres):
    w,k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::1]).reshape(-1,1)

def fracDiff_FDD(series,d,thres=1e-5):
    """
    constant window width
    """
    w,width,df=getWeights_FFD(d,thres),len(w)-1,{}
    for name in series.columns:
        seriesF,df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1 = seriesF.index[iloc1-width],seriesF[iloc1]
            if not np.isfinite(series.loc[loc1,name]): continue # exclude NAs
            df_[loc1] = np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df,axis=1)
    return df

def plotMinFFD():
    path,instName = './','ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    df0 = pd.read_csv(path+instName+'.csv',indeex_col=0,parse_dates=True)
    for d in np.linspace(0,1,11):
        df1 = np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
        df2 = fracDiff_FFD(df1,d,thres=.01)
        corr = np.corrcoef(df1.loc[df2.index,'Close'],df2['Close'])[0,1]
        df2 = adfuller(df2['Close'],maxlag=1,regression='c',autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] # with critical value
    out.to_csv(path+instName+'_testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    mpl.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    mpl.savefig(path+instName+'_testMinFFD.png')
    return

# validation

def getTrainTimes(t1,testTimes):
    """
    Given testTimes, find the times of the training observations.
    -t1.index: time when the observation started
    -t1.value: time when the observation ended
    -testTimes: times of testing observations a pandas series
    """
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0 = trn[(i<=trn.index) & (trn.index<=j)].index # train starts with test
        df1 = trn[(i<=trn) & (trn<=j)].index # train ends within test
        df2 = trn[(trn.index<=i) & (j<=trn)].index # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn

def getEmbargoTimes(times,pctEmbargo):
    """
    get embargo time for each bar
    """
    step = int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg = pd.Series(times,index=times)
    else:
        mbrg = pd.Series(times[step:],index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1],index=times[-step:]))
    return mbrg

class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping the test-label intervals
    Test set is assumed contigous (shuffle=False), w/o training samples in between
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indicies = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0 = self.t1.index[i] # start of test set
            test_indicies = indicies[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indicies].max())
            train_indicies = self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train with embargo
                train_indicies = np.concatenate((train_indicies,indicies[maxT1Idx+mbrg:]))
            yield train_indicies,test_indicies

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('Wrong scoring method')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score = []
    for train,test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train,:],y=y.iloc[train],sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test,:])
            score_ = -log_loss(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
        else:
            prob = fit.predict_proba(X.iloc[test,:])
            score_ = accuracy_score(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)

# feature importance

def featImpMDI(fit,featNames):
    """
    feat importance based on IS mean impurity reduction
    """
    df0 = {i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0,orient='index')
    df0.columns = featNames
    df0 = df0.replace(0,np.nan) # because max_features=1
    imp = pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1)
    imp /= imp['mean'].sum()
    return imp

def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    """
    feat importance based on OOS score reduction
    """
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('Wrong scoring method')
    cvGen = PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    scr0,scr1 = pd.Series(),pd.DataFrame(columns=X.columns)
    for i, (train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0 = X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1 = X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit = clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
        else:
            prob = fit.predict_proba(X1)
            scr0.loc[i] = accuracy_score(y1,prob,sample_weight=w1.iloc[test].values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i,j] = -log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            else:
                prob = fit.predict_proba(X1_)
                scr1.loc[i.j] = accuracy_score(y1,prob,sample_weight=w1.iloc[test].values)
    imp = (-scr1).add(scr0,axis=0)
    if scoring == 'neg_log_loss':imp = imp/-scr1
    else:imp=imp/(1.-scr1)
    imp = pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1)
    return imp,scr0.mean()

def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen):
    """
    single feat importance
    """
    imp = pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0 = cvScore(clf,X=trnsX[[featname]],y=cont['bin'],sample_weight=cont['w'],scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean'] = df0.mean()
        imp.loc[featName,'std'] = df0.std()*df0.shape[0]**-.5
    return imp

def get_eVec(dot,varThres):
    """
    compute eVec from dot prod matrix, reduce dimension
    """
    eVal,eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1] # args for sorting eVal
    eVal,eVec = eVal[idx],eVec[:,idx]
    # only positive eVals
    eVal = pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec,index=dot.index,columns=eVal.index)
    eVec = eVec.loc[:,eVal.index]
    # dim reduction
    cumVar = eVal.cumsum()/eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal,eVec = eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec

def orthFeats(dfX,varThres=.95):
    """
    Given a datafram dfX of features, compute orthogonal features dfP
    """
    dfZ = dfX.sub(dfX.mean(),axis=1).div(dfX.std(),axis=1) # standardise
    dot = pd.DataFrame(np.dot(dfZ.T,dfZ),index=dfX.columns,columns=dfX.columns)
    eVal,eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ,eVec)
    return dfP

def getTestData(n_features=40,n_informative=10,n_redundant=10,n_samples=10000):
    """
    Generate a random dataset for a classification problem
    """
    trnsX,cont = make_classification(n_samples=n_samples,n_features=n_features,n_informative=n_informative,
                                     n_redundant=n_redundant,random_state=0,shuffle=False)
    df0 = pd.DatetimeIndex(periods=n_samples,freq=pd.tseries.offsets.BDay(),end=pd.datetime.today())
    trnsX,cont = pd.DataFrame(trnsX,index=df0),pd.Series(cont,index=df0).to_frame('bin')
    df0 = ['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0 += ['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns = df0
    cont['w'] = 1./cont.shape[0]
    cont['t1'] = pd.Series(cont.index,index=cont.index)
    return trnsX,cont


def featImportance(trnsX,cont,n_estimators=1000,cv=10,max_samples=1.,numThreads=4,
                   pctEmbargo=0,scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    """
    feature importance from a random forest
    """
    n_jobs = (-1 if numThreads>1 else 1)  # run 1 thread with ht_helper in dirac1
    # prepare classifier
    clf = DecisionTreeClassifier(criterion='entropy',max_features=1,class_weight='balanced',
                                 min_weight_fraction_leaf=minWLeaf)
    clf = BaggingClassifier(base_estimator=clf,n_estimators=n_estimators,max_features=1.,
                            max_samples=max_samples,oob_score=True,n_jobs=n_jobs)
    fit = clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values)
    oob = fit.oob_score_
    if method == 'MDI':
        imp = featImpMDI(fit,featNames=trnsX.columns)
        oos = cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],
                      t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean()
    elif method == 'MDA':
        imp,oos = featImpMDA(clf,X=trnsX,y=cont['t1'],cv=cv,sample_weight=cont['w'],
                             t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo)
        oos = cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],
                      scoring=scoring,cvGen=cvGen).mean()
        clf.n_jobs = 1 # paralellise auxFeatImpSFI rather than clf
        imp = mpPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,
                          clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
    return imp,oob,oos

def testFunc(n_features=40,n_informative=10,n_redundant=10,n_estimators=1000,
             n_samples=10000,cv=10):
    """
    test the perdormance of the feat importance on artificial data
    Nr noise features = n_features-n_informative-n_redundant
    """
    trnsX,cont = getTestData(n_features,n_informative,n_redundant,n_samples)
    dict0 = {'minWLeaf':[0.],'scoring':['accuracy'],'method':['MDI','MDA','SFI'],
             'max_samples':[1.]}
    jobs,out = (dict(zip(dict0,i)) for i in product(*dict0.values())),[]
    kargs={'pathOut':'./testFunc','n_estimators':n_estimators,'tag':'testFunc','cv':cv}
    for job in jobs:
        job['simNum'] = job['method']+'_'+job['scoring']+'_'+'%.2f'%job['minWLeaf']+'_'+str(job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp,oob,oos = featImportance(trnsX=trnsX,cont=cont,**kargs)
        plotFeatImportance(imp=imp,oob=oob,oos=oos,**kargs)
        df0 = imp[['mean']]/imp['mean'].abs().sum()
        df0['type'] = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob':oob,'oos':oos});df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(['method','scoring','minWleaf','max_samples'])
    out = out['method','scoring','minWLeaf','max_samples','I','R','N','oob','oos']
    out.to_csv(kargs['pathOut']+'stats.csv')
    return

def plotFeatImportance(pathOut,imp,oob,oos,method,tag=0,simNum=0,**kargs):
    """
    plot mean imp bars with std
    """
    mpl.figure(figsize=(10,imp.shape[0]/5.))
    imp = imp.sort_values('mean',ascending=True)
    ax = imp['mean'].plot(kind='barh',color='b',alpha=.25,xerr=imp['std'],
                          error_kw={'ecolor':'r'})
    if method == 'MDI':
        mpl.xlim([0,imp.sum(axis=1).max()])
        mpl.axvline(1./imp.shape[0],linewidth=1,color='r',linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i,j in zip(ax.patches,imp.index):ax.text(i.get_width()/2,i.get_y()+i.get_height()/2,
                                                 j,ha='center',va='center',color='black')

    mpl.title('tag='+tag+' | simNum='+simNum+' | oob='+str(round(oob,4))+' | oos='+str(round(oos,4)))
    mpl.savefig(pathOut+'featImportance_'+str(simNum)+'.png',dpi=100)
    mpl.clf();mpl.close()
    return

# hyperparameter tuning with cross validation 

class MyPipeline(Pipeline):
    def fit(self,X,y,sample_weight=None,**fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
        return super(MyPipeline,self).fit(X,y,**fit_params)

def clfHyperFitRand(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.],
                    rndSearchIter=0,n_jobs=-1,pctEmbargo=0,**fit_params):
    """
    randomised search with purged k fold cross val
    """
    if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labelling
    else: scoring='neg_log_loss' # symmetric to all cases
    # 1) hyper parameter search on training data
    inner_cv = PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    if rndSearchIter==0:
        gs = GridSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,
                          cv=inner_cv,n_jobs=n_jobs,iid=False)
    else:
        gs = RandomSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,
                          cv=inner_cv,n_jobs=n_jobs,iid=False,n_iter=rndSearchIter)
    gs = gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps),n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
        gs = gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs=Pipeline(['bag',gs])
    return gs

# backtesting

# bet sizing

def getSignal(events,stepSize,prob,numClasses,numThreads,**kargs):
    """
    get signals from predictions
    """
    if prob.shape[0]==0:return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest,OvR)
    signal0 = (prob-1./numClasses)/(prob*(1-prob))**.5 # t value of OvR
    signal0 = pred*(2*norm.cdf(signal0)-1) # signal=side*size
    if 'side' in events:signal0*=events.loc[signal0.index,'side'] # meta-labelling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']],how='left')
    df0 = avgActiveSignals(df0,numThreads)
    signal1 = discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

def avgActiveSignals(signals,numThreads):
    """
    compute the average signal among those active
    """
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts);tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals,('molecule',tPnts),numThreads,signals=signals)
    return out

def mpAvgActiveSignals(signals,molecule):
    """
    At time loc, average signal among those still active
    signal is active if:
        issued before or at loc AND
        loc before signals endtime, or endtime is unknown NaT
    """
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values<=loc)&((loc<singals['t1'])|pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act)>0:out[loc]=signals.loc[act,'signal'].mean()
        else: out[act]=0 # no signals active at this time
    return out

def discreteSignal(signal0,stepSize):
    """
    discretise signal
    """
    signal1 = (signal0/stepSize).round()*stepSize #discretise
    signal1[signal1>1] = 1 # cap
    signal1[signal1<0] = 0 # floor
    return signal1

# dynamic position size and limit price (using sigmoid function)

def betSize(w,x):
    return x*(w+x**2)**-.5

def getTpos(w,f,mP,maxPos):
    return int(betSize(w,f-mP)*maxPos)

def invPrice(f,w,m):
    return f-m*(w/(1-m**2))**.5

def limitPrice(tPos,pos,f,w,maxPos):
    sgn = (1 if tPos>=pos else -1)
    lP = 0
    for j in range(abs(pos+sgn),abs(tPos+1)):
        lP+=invPrice(f,w,j/float(maxPos))
    lP /= tPos-pos
    return lP

def getW(x,m):
    # 0 < alpha < 1
    return x**2*(m**-2-1)

# optimal trading rules

def batch(coeffs,nIter=1e5,maxHP=100,rPT=np.linspace(.5,10,20),
         rSLm=np.linspace(.5,10,20),seed=0):
    phi,output1 = 2**(-1./coeffs['hl']), []
    for comb_ in product(rPT,rSLm):
        output2 = []
        for iter_ in range(int(nIter)):
            p,hp,count = seed,0,0
            while True:
                p = (1-phi)*coeffs['forecast']+phi*p+coeffs['sigma']*gauss(0,1)
                cP = p-seed;hp+=1
                if cP>comb_[0] or cP<-comb_[1] or hp>maxHP:
                    output2.append(cP)
                    break
        mean,std = np.mean(output2),np.std(output2)
        print(comb_[0],comb_[1],mean,std,mean/std)
        output1.append((comb_[0],comb_[1],mean,std,mean/std))
    return output1

# backtest stats

def getHoldingPeroid(tPos):
    """
    Derive avg holding period (in days) using avg entry time pairing algo
    """
    hp,tEntry = pd.DataFrame(columns=['dT','w']),0.
    pDiff,tDiff = tPos.diff(),(tPos.index-tPos.index[0])/np.timedelta64(1,'D')
    for i in range(1,tPos.shape[0]):
        if pDiff.iloc[i]*tPos.iloc[i-1]>=0: # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry*tPos.iloc[i-1]+tDiff[i]*pDiff.iloc[i])/tPos.iloc[i]
        else: # decreased
            if tPos.iloc[i]*tPos.iloc[i-1]<0: # flip
                hp.loc[tPos.index[i],['dT','w']] = (tDiff[i]-tEntry,abs(tPos.iloc[i-1]))
                tEntry = tDiff[i] # reset entry time
            else:
                hp.loc[tPos.index[i],['dT','w']] = (tDiff[i]-tEntry,abs(pDiff.iloc[i]))
    if hp['w'].sum()>0:hp=(hp['dT']*hp['w']).sum()/hp['w'].sum()
    else: hp=np.nan
    return hp

def getHHI(betRet):
    if betRet.shape[0]<=2:return np.nan
    wght = betRet/betRet.sum()
    hhi = (wght**2).sum()
    hhi = (hhi-betRet.shape[0]**-1)/(1.-betRet.shape[0]**-1)
    return hhi

def computeDD_TuW(series,dollars=False):
    """
    compute series of drawdowns and time under water because of them
    """
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hmw').min().reset_index()
    df1.columns = ['hmw','min']
    df1.index = df0['hmw'].dropduplicate(keep='first').index # time of hmw
    df1 = df1[df1['hmw']>df1['min']] # hmw followed by a drawdown
    if dollars: dd=df1['hmw']-df1['min']
    else: dd = 1-df1['min']/df1['hmw']
    tuw = ((df1.index[1:]-df1.index[:-1])/np.timedlta64(1,'Y')).values # in years
    tuw = pd.Series(tuw,index=df1.index[:-1])
    return dd,tuw

def binHR(sl,pt,freq,tSR):
    """
    Given a trading rule charicterised by the parameters {sl,pt,freq},
    what's the min precision p reqired to achieve a Sharpe ratio tSR?
    sl: stop loss threshold
    pt: profit taking threshold
    freq: number of bets per year
    tSR: target annual Sharpe Ratio

    p: the min precision rate p required to achieve tSR
    """
    a = (freq+tSR**2)*(pt-sl)**2
    b = (2*freq*sl-tSR**2*(pt-sl))*(pt-sl)
    c = freq*sl**2
    p = (-b+(b**2-4*a*c)**.5)/(2.*a)
    return p

def binFreq(sl,pt,p,tSR):
    """
    Given a trading rule characterised by the parameters {sl,pt,freq},
    what's the number of bets per tear needed to achieve a Sharpe ratio
    tSR with precision p?
    sl: stop loss threshold
    pt: profit taking threshold
    p: the precision rate p 
    tSR: target annual Sharpe Ratio

    freq: number of bets per year needed
    """
    freq = (tSR*(pt-sl))**2*p*(1-p)/((pt-sl)*p+sl)**2 # possible ectraneous
    if not np.isclose(binSR(sl,pt,freq,p),tSR):return
    return freq


def getQuasiDiag(link):
    """
    sort clustered items by distance. 
    link = linkage matrix
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1,0],link[-1,1]])
    numItems = link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index = range(0,sortIx.shape[0]*2,2) # make space
        df0 = sortIx[sortIx>=numItems] # find clusters
        i = df0.index; j = df0.values-numItems
        sortIx[i] = link[j,0] # item 1
        df0 = pd.Series(link[j,1],index=i+1)
        sortIx = sortIx.append(df0) # item 2
        sortIx = sortIx.sort_index() # resort
        sortIx.index = range(sortIx.shape[0]) # reindex
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    """
    compute HRP alloc
    """
    w = pd.Series(1,index=sortIx)
    cItems = [sortIx] # initialise all items in one cluster
    while len(cItems)>0:
        cItems = [i[j:k] for i in cItems for j,k in ((0,len(i)/2),(len(i)/2,l2n(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0 = cItems[i] # cluster 1
            cItems1 = cItems[i+1] # cluster 2
            cVar0 = getClusterVar(cov,cItems0)
            cVar1 = getClusterVar(cov,cItems1)
            alpha = 1- cVar0/(cVar0+cVar1)
            w[cItems0] *= alpha # weight 1
            w[cItems1] *= 1-alpha # weight 2
    return w                                   

def getIVP(cov,**kargs):
    """
    compute the inverse-variance portfolio
    """
    ivp = 1./np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    """
    compute variance per cluster
    """
    cov_ = cov.loc[cItems,cItems] # matrix slice
    w_ = getIVP(cov_).reshape(-1,1)
    cVar = np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

# useful financial features

def get_bsadf(logP,minSL,constant,lags):
    """
    SADF's inner loop

    logP = pandas series of log prices
    minSL minimum sample length tau, used ny the final regression
    constant = the regressions time trend component
        -'nc' = no time trend only a constant
        -'ct' = constant plus linear time trend
        -'ctt' = constant plus a second degree polynomial time trend
    lags = the number of lags used in the ADF specification
    """
    y,x = getYX(logP,constant=constant,lags=lags)
    startPoints,bsadf,allADF = range(0,y.shape[0]+lags-minSL+1),None,[]
    for start in startPoints:
        y_,x_ = y[start:],x[start:]
        bMean_,bStd_ = getBetas(y_,x_)
        bMean_,bStd_ = bMean_[0,0],bStd_[0,0]**.5
        allADF.append(bMean_/bStd_)
        if allADF[-1]>bsadf:bsadf=allADF[-1]
    out = {'Time':logP.index[-1],'gsadf':bsadf}
    return out

def getYX(series,constant,lags):
    """
    Prepare data
    """
    series_ = series.diff().dropna()
    x = lagDF(series_,lags).dropna()
    x.iloc[:,0] = series.values[-x.shape[0]-1:-1,0] # lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x,np.ones((x.shape[0],1)),axis=1)
        if constant[:2]=='ct':
            trend = np.arange(x.shape[0]).reshape(-1,1)
            x = np.append(x,trend,axis=1)
        if constant=='ctt':
            x = np.append(x,trend**2,axis=1)
    return y,x

def lagDF(df0,lags):
    df1 = pd.DataFrame()
    if isinstance(lags,int):lags=range(lags+1)
    else:lags=[int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i)+'_'+str(lag) for i in df_.columns]
        df1 = df1.join(df_,how='outer')
    return df1

def getBetas(y,x):
    xy = np.dot(x.T,y)
    xx = np.dot(x.T,x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv,xy)
    err = y - np.dot(x,bMean)
    bVar = np.dot(err.T,err)/(x.shape[0]-x.shape[1])*xxinv
    return bMean,bVar

# entropy

def plugIn(msg,w):
    """
    compute the prob mass function fo a one-dim discrete rv
    """
    pmf = pmf1(msg,w)
    out = -sum([pmf[i]*np.log2(pmf[i]) for i in pmf])/w
    return out,pmf

def pmf1(msg,w):
    """
    Compute the prob mass function for a one-dim rv
    len(msg)-w occurences
    """
    lib = {}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    for i in range(w,len(msg)):
        msg_ = msg[i-w:i]
        if msg_ not in lib:lib[msg_]=[i-w]
        else: lib[msg_]=lib[msg_]+[i-w]
    pmf = float(len(msg)-w)
    pmf = {i:len(lib[i])/pmf for i in lib}
    return pmf

def lemelZiv_lib(msg):
    """
    A library built using the LZ algorithm
    """
    i,lib = 1,[msg[0]]
    while i<len(msg):
        for j in range(i,len(msg)):
            msg_ = msg[i:j+1]
            if msg_ not in lib:
                lib.append(msg_)
                break
            i = j+1
    return lib

def matchLength(msg,i,n):
    """
    Maximum matched length+1, with overlap
    i>=n & len(msg) >= i+n
    """
    subS = ''
    for l in range(n):
        msg1 = msg[i:i+1+1]
        for j in range(i-n,i):
            msg0 = msg[j:j+1+1]
            if msg1==msg0:
                subS = msg1
                break # search for hihger one
    return len(subS)+1,subS # matched length + 1

def konto(msg,window=None):
    """
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window)
    Inverse of the avg length of the shortest non-redundant substring
    If non-redundant substrings are short, the text is highly entropic
    window==None for expanding window, in which case len(msg)%2==0
    If the end of msg is more relevant, try konto[::-1]
    """
    out = {'num':0,'sum':0,'subS':[]}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    if window is None:
        points = range(1,len(msg)/2+1)
    else:
        window = min(window,len(msg)/2)
        points = range(window,len(msg)-window+1)
    for i in points:
        if window is None:
            l,msg_ = matchLength(msg,i,i)
            out['sum'] += np.log2(i+1)/1 # to avoid Doeblin condition
        else:
            l,msg_ = matchLength(msg,i,window)
            out['sum'] += np.log2(window+1)/1 # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] +=1
    out['h'] = out['sum']/out['num']
    out['r'] = 1-out['h']/np.log2(len(msg)) # redundancy, 0<=r<=1
    return out

# corwin-schultz

def getBeta(series,sl):
    hl = series[['High','Low']].values
    hl = np.log(hl[:,0]/hl[:,1])**2
    hl = pd.Series(hl,index=series.index)
    beta = pd.stats.moments.rolling_sum(hl,window=2)
    beta = pd.stats.moments.rolling_mean(hl,window=sl)
    return beta.dropna()

def getGamma(series):
    h2 = pd.stats.moments.rolling_max(series['High'],window=2)
    l2 = pd.stats.moments.rolling_min(series['Low'],window=2)
    gamma = np.log(h2.values/l2.values)**2
    gamma = pd.Series(gamma,index=h2.index)
    return gamma.dropna()

def getAlpha(beta,gamma):
    den = 3-2*2**.5
    alpha = (2**.5-1)*(beta**.5)/den
    alpha -= (gamma/den)**.5
    alpha[alpha<0] = 0 # set negative alphas to 0
    return alpha.dropna()

def corwinSchultz(series,sl=1):
    """
    S<0 iif alpha<0
    """
    beta = getBeta(series,sl)
    gamma = getGamma(series)
    alpha = getAlpha(beta,gamma)
    spread = 2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    startTime = pd.Series(series.index[0:speead.shape[0]],index=spread.index)
    spread = pd.concat([spread,startTime],axis=1)
    spread.columns = ['Spread','Start_Time'] # 1st loc used to compute beta
    return spread

def getSigma(beta,gamma):
    """
    Estimating volatility for high-low prices
    """
    k2 = (8/np.pi)**.5
    den = 3-2*2**.5
    sigma = (2**-.5-1)*beta**.5/(k2*den)
    sigma += (gamma/k2**2*den)**.5
    sigma[sigma<0] = 0
    return sigma

# hpc
import copyreg,types
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = cls.im_class
    return _unpickle_method,(func_name,obj,cls)

def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try: func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)

copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)

def linParts(numAtoms,numThreads):
    """
    partition of atoms within a single loop
    """
    parts = np.linspace(0,numAtoms,min(numAtoms,numThreads)+1)
    parts = np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    """
    Partition of atoms with an inner loop
    """
    parts,numThreads_ = [0],min(numAtoms,numThreads)
    for num in range(numThreads_):
        part = 1+4*(parts[-1]**2+parts[-1]+numAtoms+(numAtoms+1.)/numThreads_)
        part = (-1+part**.5)/2
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang: # first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]),parts)
    return parts

def mpPandasObj(func,pdObj,numThreads=4,mpBatches=1,linMols=True,**kargs):
    """
    Parallelise jobs, return a DataFrame or series

    func = function to be parallelised, returns DataFrame
    pdObj[0] = Name of argument to pass to molecule
    pdObj[1] = List of atoms used to pass molecule
    kargs = any other argument needed by the function
    """
    if linMols: parts = linParts(len(pdObj[1]),numThreads*mpBatches)
    else: parts = nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs = []
    for i in range(1,len(parts)):
        job = {pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else:out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    return df0.sort_index()

def processJobs_(jobs):
    """
    Run jobs sequentially, for debugging
    """
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum,numJobs,time0,task):
    """
    Report progress as asynch jobs are completed
    """
    msg = [float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=4):
    """
    Run in parallel
    jobs must contain a 'func' callback for callBack
    """
    if task is None:task=jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs,out,time0 = pool.imap_unordered(expandCall,jobs),[],time.time()
    # process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leeks
    return out

def expandCall(kargs):
    """
    Expand the arguments of the call back function
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out

def processJobsRedux(jobs,task=None,numThreads=4,redux=None,reduxArgs={},
                     reduxInPlace=False):
    """
    Run in parallel
    jobs must contain a 'func' callback, for expandedCall
    redux prevents wasting memort by reducing output on the fly

    redux = callback function that carries out the reduction.
            eg redux=pd.DataFrame.add if output dataframes 
            should be summed
    reduxArgs = Dict that contains that must be passed to
                redux
    reduxInPlace = A boolean whether redux should happen
                   in place or not
    """
    if task is None:task=jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    imap,out,time0 = pool.imap_unordered(expandCall,jobs),[],time.time()
    # process asynchronous output, report progress
    for i,out_ in enumerate(imap,1):
        if out is None:
            if redux is None:out,redux,reduxInPlace=[out_],None,time.time()
            else:out=copy.deepcopy(out_)
        else:
            if reduxInPlace:redux(out,out_,**reduxArgs)
            else:out=redux(out,out_,**reduxArgs)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leeks
    if isinstance(out,(pd.Series,pd.DataFrame)):out=out.sort_index()
    return out

def mpJobList(func,argList,numThreads=4,mpBatches=1,linMols=True,
              redux=None,reduxArgs={},reduxInPlace=False,**kargs):
    """
    Improvement to mpPandasObj to use on the fly output reduction
    """
    if linMols: parts = linParts(len(pdObj[1]),numThreads*mpBatches)
    else: parts = nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs = []
    for i in range(1,len(parts)):
        job = {pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    out = processJobsRedux(jobs,redux=redux,reduxArgs=reduxArgs,
                           reduxInPlace=reduxInPlace,numThreads=numThreads)
    return out


def tick_bar(df, m):
    return df.iloc[::m]

def volume_bar(df, m):
    aux = df.reset_index()    
    idx = []
    vol_acum = []
    c_v = 0
    for i, v in aux.vol.items():
        c_v = c_v + v 
        if c_v >= m:
            idx.append(i)
            vol_acum.append(c_v)
            c_v = 0
    volume_bar = aux.loc[idx]
    volume_bar.loc[idx, 'cum_vol'] = vol_acum 
    volume_bar = volume_bar.set_index('date')
    return volume_bar

def dollar_bar(df, m):
    aux = df.reset_index()    
    idx = []
    d_acum = []
    c_dv = 0
    for i, dv in aux.dollar_vol.items():
        c_dv = c_dv + dv 
        if c_dv >= m:
            idx.append(i)
            d_acum.append(c_dv)
            c_dv = 0 
    dollar_bar = aux.loc[idx]
    dollar_bar.loc[idx, 'cum_dollar_vol'] = d_acum 
    dollar_bar = dollar_bar.set_index('date')
    return dollar_bar

def volume_bar_cum(df, m):
    aux = df.reset_index()
    cum_v = aux.vol.cumsum()  
    th = m
    idx = []
    for i, c_v in cum_v.items():
        if c_v >= th:
            th = th + m
            idx.append(i)
    return aux.loc[idx].set_index('date')

def dollar_bar_cum(df, m):
    aux = df.reset_index()
    cum_dv = aux.dollar_vol.cumsum()  
    th = m
    idx = []
    for i, c_dv in cum_dv.items():
        if c_dv >= th:
            th = th + m
            idx.append(i)
    return aux.loc[idx].set_index('date')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import TensorBoard


if __name__=="__main__":
    raw_data_path = './IVE_tickbidask.txt'

    df = pd.read_csv(raw_data_path,
                         header=None,
                         names=['day', 'time', 'price', 'bid', 'ask', 'vol'])
    df['date'] = pd.to_datetime(df['day'] + df['time'],
                                format='%m/%d/%Y%H:%M:%S')
    df['dollar_vol'] = df['price']*df['vol']
    df = df.set_index('date')
    df = df.drop(['day', 'time'],
                    axis=1)
    df = df.drop_duplicates(keep='first')



    m = 100_000
    dollar_df = dollar_bar(df, m)

    #s_date='2015-06-10 8:00:00'
    #e_date='2019-06-10 17:00:00'

    dollar_df = dollar_df.loc[~dollar_df.index.duplicated(keep='first')]
    #dollar_df = dollar_df.loc[s_date:e_date]

    dailyVol = getDailyVol(dollar_df.price)
    
    tEvents = getTEvents(dollar_df.price,0.05)
    t1 = dollar_df.price.index.searchsorted(tEvents+pd.Timedelta(days=1))
    t1 = t1[t1<dollar_df.price.shape[0]]
    t1 = pd.Series(dollar_df.price.index[t1],index=tEvents[:t1.shape[0]])

    mov9,mov12 = dollar_df.price.rolling(9).mean(),dollar_df.price.rolling(12).mean()
    side = pd.Series([1 if mov9.loc[i]>mov12.loc[i] else -1 for i in dollar_df.index],index=dollar_df.index)

    mp.freeze_support()
    events = getEvents(close=dollar_df.price,tEvents=tEvents,ptSl=[0,2],trgt=dailyVol,minRet=0.01,numThreads=4,t1=t1,side=None)
    print([dollar_df.price.loc[:dollar_df.index[100]][-5:0]])
    x = np.array([[[dollar_df.price.loc[:i][-5:-1]],dailyVol.loc[i]] for i in events.index])
    trn_ind = np.random.choice(x.shape[0],round(x.shape[0]*2/3))
    strn_ind = set(trn_ind)
    test_ind = [i for i in range(0,x.shape[0]) if i not in strn_ind]
    x_train = x[trn_ind]
    x_test = x[test_ind]

    logret = np.log(dollar_df.price.loc[events.index]).diff()
    y = np.array([1 if i>0 else 0 for i in logret])
    y_train = y[trn_ind]
    y_test = y[test_ind] 


    model = Sequential()
    model.add(Dense(64, input_shape=(5,1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    model.fit(x_train, y_train,
              verbose=1,
              epochs=20,
              batch_size=128,
              callbacks=[tensorboard])
    score = model.evaluate(x_test, y_test, batch_size=128)
    



    events['t1'] = t1
    events['side'] = side

    bins = getBins(events,dollar_df.price)
 
    events['bin'] = bins['bin']
    events['ret'] = bins['ret']

    events = dropLabels(events)

    
    
    coEvents = mpNumCoEvents(dollar_df.index,events['t1'],dollar_df.index)
    
  

    out = mpPandasObj(mpSampleW,('molecule',events.index),4,t1=events['t1'],numCoEvents=coEvents,close=dollar_df.price)
    out = out*getTimeDecay(out)
    out *= out.shape[0]/out.sum()

    clf = DecisionTreeClassifier()
    
    x = pd.DataFrame([mov9,mov12,dailyVol])

    
    cvScore(clf,x,[events['ret']>0.01],out,t1=events['t1'],cv=5)
    
  

    