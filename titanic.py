import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import time

pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 25)

train = pd.read_csv('train.csv', index_col='PassengerId')

############


#train['GID'] = 'none'

#ticket_count = train.Ticket.value_counts()
#train['ticket_count'] = train.Ticket.apply(lambda x: ticket_count.get(x, 0))

#fare_count   = train.Fare.value_counts()
#train['fare_count']   = train.Fare.apply(lambda x: fare_count.get(x, 0))

train['Last_Name']   = train.Name.str.extract(r'([A-Za-z]+)\,')
train['family_size'] = train.Parch + train.SibSp
#train['family_code'] = train['Last_Name'] + train['family_size'].astype(str)

#train['Title']= train.Name.str.extract(r' ([A-Za-z]+)\.')

fs_scale = StandardScaler()
fs_scale.fit(train['family_size'].values.reshape((-1,1)))

def get_ticket_fate(row):
    same_ticket  = train.Ticket == row.Ticket
    not_this_row = train.index != row.name
    df = train[same_ticket & not_this_row]
    if len(df) == 0:
        return 0
    elif df.Survived.max() == 1:
        return 1
    elif df.Survived.min() == 0:
        return -1

def add_family_fate(row):
    same_family  = train.family_code == row.family_code
    not_this_row = train.index != row.name
    df = train[same_family & not_this_row]
    if row.Fate != 0:
        return row.Fate
    else:
        if len(df) == 0:
            return row.Fate
        elif df.Survived.max() == 1:
            return 1
        elif df.Survived.min() == 0:
            return -1

def cabin_recorded(row):
    if pd.isna(row.Cabin):
        return -1
    else:
        return 1

def is_kid(row, thresh=14):
    if row.Age < thresh:
        return 1
    else:
        return -1

def lone_mr(row):
    if (row.Fate == 0) & (row.Title == 'Mr'):
        return -1

    


#train['ticket_fate'] = train.apply(get_ticket_fate, axis=1)


#has_family = train['family_size'] > 0
#train.loc[has_family, 'GID'] = train.loc[has_family, 'Last_Name']+\
#                              train.loc[has_family, 'family_size'].astype(str)
    
fare_bins = [-1, 25,50,1000]
train['Fare'].fillna(train['Fare'].median(), inplace=True)
fare_binner = LabelEncoder()
train['fare_bins'] = pd.cut(train.Fare, bins=fare_bins)
fare_binner.fit(train['fare_bins'])
train['fare_bins'] = fare_binner.transform(train['fare_bins'])
fare_bin_scaler = StandardScaler()
fare_bin_scaler.fit(train['fare_bins'].values.reshape((-1,1)))




'''
died_marked_lived
Pclass == 3
Sex == Female
family_size == 0
fate == 0
deck == C
'''
    




##########

def clean_preprocess(dataframe, train=True):
    model_data = pd.DataFrame(index=dataframe.index)
    model_data['Pclass'] = dataframe.Pclass.replace([1,2,3], [1, 0, -1])
    model_data['Sex']    = dataframe.Sex.replace(['male', 'female'], [-1, 1])

    dataframe['family_size'] = dataframe.Parch + dataframe.SibSp
    model_data['family_size'] = fs_scale.transform(dataframe['family_size']\
                                                   .values.reshape((-1,1)))
    
    dataframe['Last_Name']   = dataframe.Name.str.extract(r'([A-Za-z]+)\,')
    dataframe['family_code'] = dataframe['Last_Name'] + dataframe['family_size'].astype(str)
    #dataframe['Title']       = dataframe.Name.str.extract(r' ([A-Za-z]+)\.')

    dataframe['Fate']    = dataframe.apply(get_ticket_fate, axis=1)
    dataframe['Fate']    = dataframe.apply(add_family_fate, axis=1)
    #dataframe['Fate']    = dataframe.apply(lone_mr, axis=1)
    model_data['Fate']   = dataframe['Fate']


    dataframe['Fare'].fillna(dataframe['Fare'].median(), inplace=True)
    dataframe['fare_bins'] = pd.cut(dataframe.Fare, bins=fare_bins)
    dataframe['fare_bins'] = fare_binner.transform(dataframe['fare_bins'])
    model_data['fare_bins'] = fare_bin_scaler.transform(dataframe['fare_bins'].values.reshape((-1,1)))

    #dataframe['Age'].fillna(dataframe['Age'].median(), inplace=True)
    #model_data['is_kid'] = dataframe.apply(is_kid, axis=1)
    
    #model_data['Cabin_Recorded'] = dataframe.apply(cabin_recorded, axis=1)


    #dataframe.apply(lambda x: lone_poor_female(x, model_data), axis=1)

    return model_data, dataframe['Survived']

def additional_weights(X, dataframe):
    pred_live   = [1,   1,     0,    -1,  -.5]
    young       = dataframe.Age < 12.5
    male        = dataframe.Sex == 'male'
    good_fate   = dataframe.Fate >= 1
    X[young&male&good_fate] = pred_live
    
        
    


X_train, y_train = clean_preprocess(train)
print(X_train.head())


#leaf_size=5, n_neighbors=20
knn = KNeighborsClassifier(leaf_size=5, n_neighbors=20)
knn.fit(X_train, y_train)

xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)



test  = pd.read_csv('test.csv',  index_col='PassengerId')
test['Name'] = test.Name.str.replace('"','')

tc    = pd.read_csv('titanic_complete.csv')
tc.rename(columns={'name':'Name'}, inplace=True)
tc['Name'] = tc.Name.str.replace('"','')

def join(row):
    n = (tc.Name == row.Name)
    t = (tc.ticket == row.Ticket)
    s = (tc.sex == row.Sex)
                
    k = tc[n & t & s]
    if len(k) == 1:
        return k.survived.item()

test['Survived'] = test.apply(join, axis=1)

X_test, y_test = clean_preprocess(test)
additional_weights(X_test, test)

print(X_test.head())
print('\n')
print( 'BEST: 0.8157894736842105')
print(f'KNN:  {knn.score(X_test, y_test)}')
print(f'XGBC: {xgbc.score(X_test, y_test)}')
print(f'RFC: {rfc.score(X_test, y_test)}')
print(f'LR: {lr.score(X_test, y_test)}')

knn_pred = knn.predict(X_test)

out = test[test.Survived != knn_pred].copy()

test['color'] = 'blue'
test.loc[test.index.isin(out.index), 'color'] = 'red'

test['wrong'] = [0 if x=='blue' else 1 for x in test.color]


save = pd.DataFrame(knn_pred, columns=['Survived'], index=test.index)







'''
Wrongfully died
title == miss
sex == female
Pclass == 2&3
Fare ~ 7.7

Wrongfully lived
title == Mr
sex == Male
Pclass 1&3
fare ~ 7.7
family_size = 0
'''


# extract for penalty
knn_data = X_train.copy()
knn_data['survived'] = y_train
knn_data.loc[knn_data['survived'] == 0, 'color'] = 'C3'
knn_data.loc[knn_data['survived'] == 1, 'color'] = 'C0'

def jitter(iterable, span=.2, add=False):
    if add:
        return [np.random.uniform(-span, span)+x for x in iterable]
    else:
        return [np.random.uniform(-span, span) for _ in iterable]


fig = plt.figure()

# pclass
pclass  = fig.add_subplot(151)
x       = jitter(knn_data.Pclass)
y       = jitter(knn_data.Pclass, add=True)
pclass.scatter(x, y, c=knn_data.color)
pclass.set_title('Pclass')
pclass.set_xticks(ticks=[], labels=[])

# sex
sex     = fig.add_subplot(152)
x       = jitter(knn_data.Sex)
y       = jitter(knn_data.Sex, add=True)
sex.scatter(x, y, c=knn_data.color)
sex.set_title('Sex')
sex.set_xticks(ticks=[], labels=[])

# family_size
family_size  = fig.add_subplot(153)
x           = jitter(knn_data.family_size)
y           = jitter(knn_data.family_size, add=True)
family_size.scatter(x, y, c=knn_data.color)
family_size.set_title('family_size')
family_size.set_xticks(ticks=[], labels=[])

# fate
Fate  = fig.add_subplot(154)
x           = jitter(knn_data.Fate)
y           = jitter(knn_data.Fate, add=True)
Fate.scatter(x, y, c=knn_data.color)
Fate.set_title('Fate')
Fate.set_xticks(ticks=[], labels=[])

# fare_bins
fare_bins  = fig.add_subplot(155)
x           = jitter(knn_data.fare_bins)
y           = jitter(knn_data.fare_bins, add=True)
fare_bins.scatter(x, y, c=knn_data.color)
fare_bins.set_title('fare_bins')
fare_bins.set_xticks(ticks=[], labels=[])

plt.show()

knn.fit(X_train, y_train)
additional_weights(X_train, train)

y_pred = knn.predict(X_train)
train['y_pred'] = y_pred
wrong = train[train.Survived != train.y_pred]

dml = wrong[wrong.Survived == 0]
lmd = wrong[wrong.Survived == 1]

##for i in range(len(dml.columns)):
##    print(dml.iloc[:, i].value_counts(normalize=True))
##    print('\n')

'''
died_marked_lived
Pclass == 3
Sex == Female
Parch == 0
fate == 0
fare_bins == 0
22 <= fare <= 26
'''

'''
lived_marked_died
Pclass == 3
Sex == Male
family_size == 0
fate == 0
'''
def color(row):
    if row.Survived == row.y_pred:
        return 'green'
    # lived predicted died
    elif (row.Survived == 1) & (row.y_pred == 0):
        return 'blue'
    # died predicted lived
    elif (row.Survived == 0) & (row.y_pred == 1):
        return 'red'

train['color'] = train.apply(color, axis=1)

fig = plt.figure()
alpha=.7
# pclass
pclass = fig.add_subplot(221)
x = jitter(train.Pclass)
y = jitter(train.Pclass, add=True)
pclass.scatter(x, y, c=train.color, alpha=alpha, marker='.')
pclass.set_title('Pclass')
pclass.set_xticks(ticks=[], labels=[])
pclass.set_yticks(ticks=[1,2,3])

# sex
sex = fig.add_subplot(222)
x = jitter(train.Sex)
y = jitter(train.Sex.replace(['male', 'female'], [0,1]), add=True)
sex.scatter(x, y, c=train.color, alpha=alpha, marker='.')
sex.set_title('Sex')
sex.set_xticks(ticks=[], labels=[])
sex.set_yticks(ticks=[0,1], labels=['male', 'female'],rotation=90)

# age
age = fig.add_subplot(223)
x = jitter(train.Age)
y = jitter(train.Age, add=True)
age.scatter(x, y, c=train.color, alpha=alpha, marker='.')
age.set_title('Age')
age.set_xticks(ticks=[], labels=[])

# fate
fate = fig.add_subplot(224)
x = jitter(train.Fate)
y = jitter(train.Fate, add=True)
fate.scatter(x, y, c=train.color, alpha=alpha, marker='.')
fate.set_title('Fate')
fate.set_xticks(ticks=[], labels=[])


plt.show()


young       = train.Age < 12.5
male        = train.Sex == 'male'
good_fate   = train.Fate == 1
lived       = train.Survived == 1
wrong_y     = train.Survived != train.y_pred

rich        = train.fare_bins == 2
no_kids     = train.SibSp == 0
has_spouse  = train.SibSp == 1


