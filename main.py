import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing
import ml
import activities as pre

result=get_df()
# Activitiy_01 den sonra gelebilecek aktiviteler.


activity_dict={'activity_01':['activity_02','activity_03'],'activity_02':['activity_03','activity_04'],\
               'activity_03':['activity_01','activity_04','activity_05','activity_07'],\
                   'activity_04':['activity_01','activity_04'],'activity_05':['activity_02','activity_05']}

activity_01=pre.preprocesing('activity_01', activity_dict['activity_01'])
activity_02=pre.preprocesing('activity_02',activity_dict['activity_02'])
activity_03=pre.preprocesing('activity_03',activity_dict['activity_03'])
activity_04=pre.preprocesing('activity_04',activity_dict['activity_04'])
activity_05=pre.preprocesing('activity_05',activity_dict['activity_05'])

tahmin_1=ml.Algorithms(activity_01)
