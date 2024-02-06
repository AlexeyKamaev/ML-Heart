import pandas as pd
import numpy as np


import sklearn
sklearn.set_config(transform_output="pandas")
import warnings
warnings.filterwarnings('ignore')

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

# Metrics
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from catboost import CatBoostRegressor


class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.params_ = X.groupby(pd.cut(X['RestingBP'], bins=6))['Age'].median().to_dict()
        # Сохраним в атрибуте экземпляра медианные значения возраста по интервалам RestingBP
        return self

    def transform(self, X, y=None):  # Запустится тогда, когда нужно будет отдать результат на следующий шаг

        X_copy = X.copy()
        X_copy['Age'] = X_copy['Age'].fillna(X['RestingBP'].map(self.params_))

        return pd.DataFrame(X_copy[['Age', "RestingBP"]])


class CholesterolImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        cut_bins = [-10, 80, 475, 1000]
        cut_labels_4 = [1, 0, 1.1]

        X_copy['Cholesterol'] = pd.cut(X_copy['Cholesterol'],
                                       bins=cut_bins,
                                       labels=cut_labels_4)

        return pd.DataFrame(X_copy['Cholesterol']).astype(float)


class MaxHRImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        cut_bins = [-1000, 130, 150, 3000]
        cut_labels_4 = [1, 0.5, 0]

        X_copy['MaxHR'] = pd.cut(X_copy['MaxHR'],
                                 bins=cut_bins,
                                 labels=cut_labels_4)

        return pd.DataFrame(X_copy['MaxHR']).astype(float)


class reset_index_X(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        X_copy.reset_index(inplace=True, drop=True)

        return X_copy

import joblib
import streamlit as st



xx = joblib.load('ml_pipeline.pkl')



st.title('Heart Disease detector')

st.write('choose your option. If you dont know the answer, keep the auto')

sex = st.radio(
    "Sex:",
    ["M", "F"],
    captions = [":rainbow[Male]", "***Female***"]
)

age = st.slider(
    'Select yor age',
    15.0, 80.0, (25.0))

st.write('+ RestingBP: resting blood pressure [mm Hg]')
RestingBP = st.slider(
    'Select your RestingBP',
    0.0, 300.0, (100.0))
st.write('RestingBP:', RestingBP)

st.write('+ Cholesterol: serum cholesterol [mm/dl]')
cholo = st.slider(
    'Select a range of values',
    -100.0, 1000.0, (100.0))
st.write('Cholesterol:', cholo)


st.write('+ MaxHR: maximum heart rate achieved')

hr = st.slider(
    'Select a range of values',
    0.0, 250.0, (220-age))

st.write('MaxHR:', hr)




st.write('+ Chest Pain Type:')
chp = st.radio(
    "ChestPainType:",
    ["ASY", "NAP",'TA','ATA'],
captions=['ASY: Asymptomatic','NAP: Non-Anginal Pain','TA: Typical Angina', 'ATA: Atypical Angina'])



st.write('+ FastingBS: fasting blood sugar')
fbs = st.radio(
    "FastingBS:",
    ["0", "1"],
    captions = ['0: otherwise','1: if FastingBS > 120 mg/dl' ])
if int(fbs) == 1:
    st.write('seems you have some problems with :rainbow[diabetes]')


st.write("+ RestingECG: resting electrocardiogram results ")
recg = st.radio(
    "RestingECG:",
    ["Normal", "ST",'LVH'],
captions=['Normal: Normal', 'ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 'LVH: showing probable or definite * left ventricular hypertrophy by Estes criteria'])


st.write('+ ExerciseAngina: exercise-induced angina')
ang = st.radio(
    "Exercise Angina:",
    ["Y", "N"], captions=['Yes','No'])


st.write('+ ST_Slope: the slope of the peak exercise ST segment')

st_Slope = st.radio(
    "ST Slope:",
    ["Up","Flat",'Down'])
st.write(st_Slope)

old = 0
input_data = [age,sex,chp,RestingBP,cholo,fbs,recg,hr,ang,old,st_Slope]

data = pd.DataFrame(input_data).T.rename(columns={0:'Age',1:'Sex',2:'ChestPainType',3:'RestingBP',4:'Cholesterol',
5:'FastingBS',6:'RestingECG',7:'MaxHR',8:'ExerciseAngina',9:'Oldpeak',10:'ST_Slope'})
p = xx.predict(data)

x = st.button("Get me answer", type="primary")

if x:
    if p == 1:
        st.page_link("pages/ok.py", label='click on me!', icon='🫀')
    else:
        st.page_link("pages/ok.py", label="click on me!", icon ='🫀' )