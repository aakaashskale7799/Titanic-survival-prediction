# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:33:45 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:24:54 2020

@author: User titanic from github for kaggle competition
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
titanic_df = pd.read_csv('titan_train.csv')
titanic_t = pd.read_csv('test.csv')


print("Finding missing Values")
print("===========================")
titanic=pd.DataFrame(titanic_df, index=None)
titanict=pd.DataFrame(titanic_t, index=None)
print(titanic.isnull().sum())
print("Fixing missing Values")
print("===========================")
mode=titanic["Age"].mode()
modet=titanict["Age"].mode()
modet
titanic["Age"]=titanic["Age"].fillna(24)
titanict["Age"]=titanict["Age"].fillna(24)
titanict["Age"]
print(titanict.isnull().sum())
"""
titanic["Cabin"]
mode=titanic["Cabin"].mode()
titanic["Cabin"]=titanic["Cabin"].fillna(mode)
titanic["Cabin"]
"""





titanic=titanic.drop(["Name","Cabin"], axis="columns", index=None)
titanict=titanict.drop(["Name","Cabin"], axis="columns", index=None)

titanict.columns


"""
sns.factorplot('Sex',data=titanic_df,kind='count')
titanic_df.columns
sns.factorplot('Age',data=titanic_df,kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Age',kind='count')
sns.factorplot('Sex',data=titanic_df,hue='Age',kind='count')
"""
#function to detect person is man women or child
def man_wom_chi(passenger):
    age=passenger["Age"]
    sex=passenger["Sex"]
    return "child" if age < 16 else sex
titanic['Person'] = titanic.apply(man_wom_chi,axis=1)

def man_wom_chii(passenger):
    age=passenger["Age"]
    sex=passenger["Sex"]
    return "child" if age < 16 else sex
titanict['Person'] = titanict.apply(man_wom_chii,axis=1)






titanict[0:10]
print (titanict['Person'].value_counts())
#sns.factorplot('Pclass',data=titanic,hue='Person',kind='count')
#titanic['Age'].hist()
titanict['Person'].count()
titanict['PassengerId'].count()
#class v/s age

#fig = sns.FacetGrid(titanic,hue='Pclass',aspect=4)
#fig.map(sns.kdeplot,'Age',shade=True)
#oldest = titanic['Age'].max()
#fig.set(xlim=(0,oldest))
#fig.add_legend()
#sex v/s age
#fig = sns.FacetGrid(titanic,hue='Sex',aspect=4)
#fig.map(sns.kdeplot,'Age',shade=True)
#oldest = titanic['Age'].max()
#fig.set(xlim=(0,oldest))
#fig.add_legend()
#deck_df = titanic_df.dropna(axis=0)
#deck_df.head()
#How do we find out what deck a passenger was assigned?
#We just need to create a python method to extract first character from the cabin information.
"""def get_level(passenger):
    cabin=passenger["Cabin"]
    return cabin[0]
deck_df['level']=deck_df.apply(get_level,axis=1)
"""
"""
sns.factorplot('level',data=deck_df,palette='winter_d',kind='count')

sns.factorplot('level',data=deck_df,hue='Pclass',kind='count')
sns.factorplot('Embarked',data=titanic_df,hue='Pclass',x_order=['C','Q','S'],kind='count')
"""
#who was with the family?
titanic.head()
titanic['Alone'] = titanic.SibSp + titanic.Parch
titanic.tail()
titanic['Alone'].loc[titanic['Alone']>0] = 'No'

titanic['Alone'].loc[titanic['Alone']==0] = 'Yes'



titanict.head()
titanict['Alone'] = titanict.SibSp + titanict.Parch
titanict.tail()
titanict['Alone'].loc[titanict['Alone']>0] = 'No'

titanict['Alone'].loc[titanict['Alone']==0] = 'Yes'
titanict['Alone'].count()








titanic.head()
#sns.factorplot('Pclass','Survived',data=titanic)
titanic_df.isnull().sum()



#titanic=titanic["Sex"].dropna()
#Label encoder and Transformation
PersonL=LabelEncoder()
titanic["TPerson"]=PersonL.fit_transform(titanic["Person"])
titanic=titanic.drop(["Sex"],axis="columns")
titanic=titanic.drop(["Person"],axis="columns")

AloneL=LabelEncoder()
titanic["TAlone"]=AloneL.fit_transform(titanic["Alone"])
titanic=titanic.drop(["Alone"],axis="columns")


titanic["Embarked"]=str(titanic["Embarked"])
EmbarkedL=LabelEncoder()
titanic["TEmbarked"]=AloneL.fit_transform(titanic["Embarked"])
titanic=titanic.drop(["Embarked"],axis="columns")

#==============================================================
PersonLt=LabelEncoder()
titanict["TPersont"]=PersonLt.fit_transform(titanict["Person"])
titanict=titanict.drop(["Sex"],axis="columns")
titanict=titanict.drop(["Person"],axis="columns")

AloneLt=LabelEncoder()
titanict["TAlonet"]=AloneLt.fit_transform(titanict["Alone"])
titanict=titanict.drop(["Alone"],axis="columns")


titanict["Embarked"]=str(titanict["Embarked"])
EmbarkedLt=LabelEncoder()
titanict["TEmbarkedt"]=AloneLt.fit_transform(titanict["Embarked"])
titanict=titanict.drop(["Embarked"],axis="columns")
titanict=titanict.drop(["Ticket"],axis="columns")
#====================================================================
titanict.dtypes

#applying the model
data=titanic.drop(["Survived","Ticket"],axis="columns")
datat=titanict
target=titanic["Survived"]
from sklearn.svm import SVC
model=SVC()
model.fit(data,target)
sc=model.score(data,target)



print("===========here is your accuracy score=========================")
print(sc)
data.columns
"""pre=model.predict([[PassengerId, Pclass, Age, SibSp, Parch, 
                    Fare, TPerson,
                    TAlone, TEmbarked]])
"""
data.dtypes
titanict.dtypes
c=datat["Age"]

titanic.isnull().sum()
datat.isnull().sum()


modedt=datat["Fare"].mode()
modedt
datat["Fare"]=datat["Fare"].fillna(7.75)
datat.isnull().sum()



ypred=model.predict(datat)
ypred


fpred=pd.DataFrame(ypred)
idt=pd.read_csv("test.csv")
datasets=pd.concat([idt["PassengerId"],fpred],axis=1)
datasets.columns=["PassengerId","Survived"]
datasets.to_csv("Kagglehelp.csv",index=False)



