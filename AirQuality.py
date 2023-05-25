import pandas as pd
import numpy as np
df=pd.read_csv('/home/student/Desktop/DATASETS/AirQuality.csv',encoding='cp1252',low_memory=False)

df.isnull().sum()
df.describe()
df=df.drop(['stn_code','agency','sampling_date','location_monitoring_station'], axis=1)

df.shape

df.dropna(subset=(['date']))

df.columns
#changing types to Uniform format
df['type'].unique()
types={
    "Residential":"R",
    "Residential and Others":"RO",
    "Residential, Rural and other Areas":"RRO",
    "Industrial Area":"I",
    "Industrial Areas":"I",
    "Industrial":"I",
    "Sensitive Area":"S",
    "Sensitive Areas":"S",
    "Sensitive":"S",
    "NaN":"RRO"
}

df.type=df.type.replace(types)

df.head()

df['date']=pd.to_datetime(df['date'],errors='coerce')

df.head()

COLS=['so2','no2','rspm','spm','pm2_5']

from sklearn.impute import SimpleImputer
i=SimpleImputer(missing_values=np.nan, strategy='mean')
df[COLS]=i.fit_transform(df[COLS])

df.head(5)
df['type'].unique()
df['type'].value_counts
df['type'].replace({"RRO":1,"RO":2,"I":3,"S":4,"RIROU":5,"R":6},inplace=True)
df['type']

df['state'].value_counts

from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
df['state']=lr.fit_transform(df['state'])
df.head(5)
