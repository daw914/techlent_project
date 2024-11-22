import pickle
import pandas as pd 
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
import shap


#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline as imbpipeline
#from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
#from imblearn.over_sampling import SMOTE
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
#from sklearn.model_selection import GridSearchCV, KFold
#from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('Telco.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

features = list(df.columns)
target = 'Churn'
features.remove(target)
def churn2num(x):
    if x=='Yes':
        return 1
    else:
        return 0


X= df[features]
y = df[target].map(churn2num)

class Transformer(object):
        
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['gender'] = X.gender.map(self.gender2num)
        df['SeniorCitizen'] = X.SeniorCitizen
        df['Partner'] = X.Partner.map(self.dummy2num)
        df['Dependents'] = X.Dependents.map(self.dummy2num)
        df['tenure'] = X.tenure
        df['DSL'] = X.InternetService.map(self.DSL)
        df['FiberOptic'] = X.InternetService.map(self.fiber)
        df['PhoneService'] = X.PhoneService.map(self.dummy2num)
        df['MultipleLines'] = X.MultipleLines.map(self.dummy2num)
        df['OnlineSecurity'] = X.OnlineSecurity.map(self.dummy2num)
        df['OnlineBackup'] = X.OnlineBackup.map(self.dummy2num)
        df['DeviceProtection'] = X.DeviceProtection.map(self.dummy2num)
        df['TechSupport'] = X.TechSupport.map(self.dummy2num)
        df['StreamingTV'] = X.StreamingTV.map(self.dummy2num)
        df['StreamingMovies'] = X.StreamingMovies.map(self.dummy2num)
        df['PaperlessBilling'] = X.PaperlessBilling.map(self.dummy2num)  
        df['Contract'] = X.Contract.map(self.contract2num)
        #df['Autopay']= X.PaymentMethod.map(self.autopay)
        df['ElectronicCheck']= X.PaymentMethod.map(self.ele_pay)
        df['MonthlyCharges']= X.MonthlyCharges
        df['TotalCharges']= X.TotalCharges.fillna(0)
        self.mean = df.mean()
        
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['gender'] = X.gender.map(self.gender2num)
        df['SeniorCitizen'] = X.SeniorCitizen
        df['Partner'] = X.Partner.map(self.dummy2num)
        df['Dependents'] = X.Dependents.map(self.dummy2num)
        df['tenure'] = X.tenure
        df['DSL'] = X.InternetService.map(self.DSL)
        df['FiberOptic'] = X.InternetService.map(self.fiber)
        df['PhoneService'] = X.PhoneService.map(self.dummy2num)
        df['MultipleLines'] = X.MultipleLines.map(self.dummy2num)
        df['OnlineSecurity'] = X.OnlineSecurity.map(self.dummy2num)
        df['OnlineBackup'] = X.OnlineBackup.map(self.dummy2num)
        df['DeviceProtection'] = X.DeviceProtection.map(self.dummy2num)
        df['TechSupport'] = X.TechSupport.map(self.dummy2num)
        df['StreamingTV'] = X.StreamingTV.map(self.dummy2num)
        df['StreamingMovies'] = X.StreamingMovies.map(self.dummy2num)
        df['PaperlessBilling'] = X.PaperlessBilling.map(self.dummy2num)
        df['Contract'] = X.Contract.map(self.contract2num)
        #df['Autopay']= X.PaymentMethod.map(self.autopay)
        df['ElectronicCheck']= X.PaymentMethod.map(self.ele_pay)
        df['MonthlyCharges']= X.MonthlyCharges
        df['TotalCharges']= X.TotalCharges.fillna(0)
        return df.fillna(self.mean)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def gender2num(self, x):
        if type(x) == str: 
            if x=='Female':  #female is 1
                return 1
            else:
                return 0
        else:
            return x

    def dummy2num(self, x):
        if type(x) == str: 
            if x=='Yes':  
                return 1
            else:
                return 0
        else:
            return x
        
    def DSL(self, x):
        if type(x) == str: 
            if x=='DSL':  
                return 1
            else:
                return 0
        else:
            return x
        
    def fiber(self, x):
        if type(x) == str: 
            if x=='Fiber optic':  
                return 1
            else:
                return 0
        else:
            return x

    def contract2num(self, x):
        if type(x) == str: 
            if x=='Month-to-month':  
                return 0
            elif x=='One year':
                return 1
            else:
                return 2
        else:
            return x
        
    def autopay(self, x):
        if type(x) == str: 
            if x=='Bank transfer (automatic)':  
                return 1
            elif x=='Credit card (automatic)':
                return 1
            else:
                return 0
        else:
            return x
    def ele_pay(self, x):
        if type(x) == str: 
            if x=='Electronic check':  
                return 1
            else:
                return 0
        else:
            return x

params = {'verbose':0,
         'iterations':200,
         'learning_rate':0.01,
         'depth':7}

steps=[('tf', Transformer()),
    ('cat', CatBoostClassifier(**params))]
model=Pipeline(steps)

model=model.fit(X,y)

with open('model.pkl','wb') as f:
    pickle.dump(model,f)