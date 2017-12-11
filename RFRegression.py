
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# In[ ]:

def runbagging(SampleNo):
    wsdata=pd.read_csv("WS" + str(SampleNo) + ".csv")
    wsdata = wsdata.drop(['Data_Source','AD_FREQ_NEW', 'TPD_FREQ_NEW',                           'TPPI_P_FREQ_NEW', 'TH_FREQ_NEW_NSC', 'ad_sev_new',                           'tpd_sev_new', 'th_sev_new', 'th_sev_new_nsc', 'tppi_cpc_new_nsc'], axis=1)
    wsdata = pandas.get_dummies(wsdata)
    msk = np.random.rand(len(wsdata))
    trnmsk = msk<=0.7
    vldmsk = msk>0.7
    trdt = wsdata[trnmsk]
    vldt = wsdata[vldmsk]
    trdt_X = trdt.drop(['WS_FREQ_NEW'], axis=1)
    vldt_X = vldt.drop(['WS_FREQ_NEW'], axis=1)
    trdt_Y = trdt.drop['WS_FREQ_NEW']
    vldt_Y = vldt.drop['WS_FREQ_NEW']
    trdt, vldt, wsdata = None, None, None
    del trdt, vldt, wsdata
    outDF = pd.DataFrame(columns=['SampleNumber', 'n_estimators', 'max_features', 'min_split', 'Score'])
    for i in Range(1, 6): 
        for j in Range(10, 31:
            for k in Range(1, 11):
                n = i *100
                m = j
                s = 0.01*k
                RFmodel = RandomForestRegressor(n_estimators = n, max_features = m, min_samples_split=s)
                RFmodel.fit(trdt_X, trdt_Y)
                score = RFmodel.score(vldt_X, vldt_Y)
                thisline = [SampleNo, n, m, s, score]
                outDF.loc[outDF.shape[0]] =  thisline
    return outDF


# In[ ]:

outDF_Final = pd.DataFrame(columns=['SampleNumber', 'n_estimators', 'max_features', 'min_split', 'Score'])
for i in Range(1,9):
    outDF_Final = pd.concat([outDF_Final, runbagging(i)], axis=0)
outDF_Final.to_csv("RFResults.csv")

