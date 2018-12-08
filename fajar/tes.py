# Program untuk menentukan apakah seseorang dapat menghasilkan lebih dari 50.000 dolar per tahun
# Dataset yang digunakan : https://archive.ics.uci.edu/ml/datasets/Adult
# Jumlah data : 48842
# Jumlah atribut pada dataset : 14
# Null values : yes

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Load file Dataset untuk dijadikan DataFrame
adult_df = pd.read_csv('adult.data', header = None, delimiter=' *, *', engine='python')

# Menambahkan Header pada DataFrame
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

# print(adult_df)

# Checking null values
#
# print(adult_df.isnull().sum())
# for value in ['workclass', 'education',
#           'marital_status', 'occupation',
#           'relationship','race', 'sex',
#           'native_country', 'income']:
#     print(value,":", sum(adult_df[value] == '?'))
#
# Terdapat null values yang ditandai oleh "?" pada atribut workclass, occupation, dan native_country

# Mengkopikan DF original ke sebuah variable baru
copy_of_df = adult_df

# Melihat statistik dari DataFrame yang ada
#print(copy_of_df.describe(include='all'))

for value in ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']:
    # Mengganti "?" dengan Most-Common Value dari tiap Kolom
    copy_of_df[value].replace(['?'], [copy_of_df.describe(include='all')[value][2]], inplace=True)

#convert label menjadi numerical menggunakan LabelEncoder
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat   = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)

copy_of_df['workclass_cat'] = workclass_cat
copy_of_df['education_cat'] = education_cat
copy_of_df['marital_cat'] = marital_cat
copy_of_df['occupation_cat'] = occupation_cat
copy_of_df['relationship_cat'] = relationship_cat
copy_of_df['race_cat'] = race_cat
copy_of_df['sex_cat'] = sex_cat
copy_of_df['native_country_cat'] = native_country_cat

dummy = ['workclass', 'education', 'marital_status',
                  'occupation', 'relationship', 'race',
                  'sex', 'native_country']

# Drop kolom di list dummy, agar data lebih fresh dan hanya berisikan value numerical
copy_of_df = copy_of_df.drop(dummy, axis = 1)

copy_of_df = copy_of_df.reindex_axis(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                                    'education_num', 'marital_cat', 'occupation_cat',
                                    'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                                    'capital_loss', 'hours_per_week', 'native_country_cat',
                                    'income'], axis=1)

#print(copy_of_df.head(1))

# Standardisasi Data
num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country_cat']

scaled_features = {}
for each in num_features:
    mean, std = copy_of_df[each].mean(), copy_of_df[each].std()
    scaled_features[each] = [mean, std]
    copy_of_df.loc[:, each] = (copy_of_df[each] - mean) / std

# print(copy_of_df)

features = copy_of_df.values[:,:14]
kelas = copy_of_df.values[:,14]
features_train, features_test, kelas_train, kelas_test = train_test_split(features, kelas, test_size = 0.33, random_state = 10)

print(features_test)
print(features_train)
print(kelas_test)
print(kelas_train)

clf = GaussianNB()
clf.fit(features_train, kelas_train)
predict = clf.predict(features_test)

print(accuracy_score(kelas_test, predict, normalize = True))
