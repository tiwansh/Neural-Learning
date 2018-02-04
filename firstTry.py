import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#1 Take a look at data
#print train_df.info()

#2 Describe the data
#print train_df.describe()

#3 Test the correlation
#print train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()
#print train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean()
#print train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by = 'Survived', ascending = False)
#print train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['Fare','Survived']].groupby(['Fare'], as_index = False).mean() - no insight - scattered data
#print train_df[['Age','Survived']].groupby(['Age'], as_index = False).mean() - no insight again - Scattered data

#4 Plot histograms/scatter graph/kdeplot in cases when data is fucking scattered
#grid = sns.FacetGrid(train_df, col = 'Survived');
#grid.map(plt.hist, 'Fare', bins = 20);  #histogram needs bins attribute
#grid.map(plt.scatter, 'Fare','Pclass'); #no bins in scatter 
#grid.map(sns.kdeplot, 'Pclass', 'Fare');
plt.show()