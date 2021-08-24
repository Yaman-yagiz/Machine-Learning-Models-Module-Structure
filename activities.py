import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

result=get_df()

def preprocesing(acitivity_number,events):
    act_arr=[]
    act_arr=np.array(act_arr)
    count=0
    activity=result[result['activities']==acitivity_number]
    if len(events)<3:
        for i in activity.iloc:
            if i[3]=='80yo+' and i[4]=='middle' and i[2]=='no':
                act_arr=np.append(act_arr,events[1])
            else:
                act_arr=np.append(act_arr,events[0])
        activity['target']=act_arr
        activity=activity.drop(columns=['trace_id','activities'])
        pre_result=pre(activity,events)
        return pre_result
    
    else:
        for i in activity.iloc:
            if i[3]=='80yo+' and i[4]=='middle' and i[2]=='no' and count%2==0:
                act_arr=np.append(act_arr,events[0])
            elif i[3]=='80yo+' and i[4]=='middle' and i[2]=='no' and count%2!=0:
                act_arr=np.append(act_arr,events[2])
            elif i[3]=='0-19yo' and i[4]=='low' and i[2]=='yes':
                act_arr=np.append(act_arr,events[3])
            else:
                act_arr=np.append(act_arr,events[1])
            count+=1
    
        activity['target']=act_arr
        activity=activity.drop(columns=['trace_id','activities'])
        pre_result=pre_for_multi_class(activity)
        return pre_result
        
        
def pre_for_multi_class(activity):
    events=['activity_01','activity_02','activity_03','activity_04','activity_05','*','activity_07']
    le=preprocessing.LabelEncoder()
    # veri ön işleme multi classification
    activity['target']=[events.index(i)+1 if i in events else 0 for i in activity['target'].iloc]
    target=activity.iloc[:,3:4].values
    df_target=pd.DataFrame(target,columns=['target'])
    
    age=activity.iloc[:,1:2].values
    ohe=preprocessing.OneHotEncoder()
    age=ohe.fit_transform(age).toarray()
    df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')
    
    income=activity.iloc[:,2:3].values
    ohe=preprocessing.OneHotEncoder()
    income=ohe.fit_transform(income).toarray()
    df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')
    
    employed=activity.iloc[:,0:1].values
    employed[:,0]=le.fit_transform(activity.iloc[:,0])
    df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')
    
    frames=[df_employed,df_age,df_income,df_target]
    pre_result= pd.concat(frames,axis=1)
    return pre_result

def pre(activity,events):
    # veri ön işleme binary classification
    activity['target']=np.where(activity['target']==events[0],int(events[0][-1]),int(events[1][-1]))
    target=activity.iloc[:,3:4].values
    df_target=pd.DataFrame(target,columns=['target'])
    le=preprocessing.LabelEncoder()
    
    age=activity.iloc[:,1:2].values
    ohe=preprocessing.OneHotEncoder()
    age=ohe.fit_transform(age).toarray()
    df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')
    
    income=activity.iloc[:,2:3].values
    ohe=preprocessing.OneHotEncoder()
    income=ohe.fit_transform(income).toarray()
    df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')
    
    employed=activity.iloc[:,0:1].values
    employed[:,0]=le.fit_transform(activity.iloc[:,0])
    df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')
    
    frames=[df_employed,df_age,df_income,df_target]
    pre_result = pd.concat(frames,axis=1)
    return pre_result