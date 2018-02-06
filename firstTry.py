import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#1 -----------------------Take a look at data----------------------
#print train_df.info()
#print train_df.isnull().sum()

#2 -----------------------Describe the data------------------------
#print train_df.describe()

#3 -----------------------Test the correlation---------------------
#print train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()
#print train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean()
#print train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by = 'Survived', ascending = False)
#print train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print train_df[['Fare','Survived']].groupby(['Fare'], as_index = False).mean() - no insight - scattered data
#print train_df[['Age','Survived']].groupby(['Age'], as_index = False).mean() - no insight again - Scattered data

#4 ---------------------Plot histograms/scatter graph/kdeplot in cases when data is scattered--------------
#grid = sns.FacetGrid(train_df, col = 'Survived');
#grid.map(plt.hist, 'Fare', bins = 20);  #histogram needs bins attribute
#grid.map(plt.scatter, 'Fare','Pclass'); #no bins in scatter 
#grid.map(sns.kdeplot, 'Pclass', 'Fare');
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();
#grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6);
#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep');
#grid.add_legend();
#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
#grid.add_legend()
#plt.show()

#5 --------------------------Manipulation of data-----------------------------------------
#concatenate both training and test data so that they are manipulated in a single go

data_df = pd.concat((train_df, test_df)) #.reset_index(drop = True) 
#can be used to reset the index column which if dropped using True, it will be removed from the dataframe
#print data_df.shape

data_df['Sex'] = data_df['Sex'].map({'female' : 1, 'male' : 0}).astype(int)

#print data_df.Embarked.isnull() #prints whether null or not
#print data_df.Embarked.isnull().values # Gives a series of values as false and true i.e. 0 and 1
#print data_df.Embarked.isnull().values.sum() # Gives the sum of 0s and 1s which means sum of all trues

#
freq_port = data_df.Embarked.dropna().mode()[0]
#dropna drops all the null items 
#mode return most frequent entry -> 0 S 
#and [0] is used to access the most frequent entry i.e. S

#Filling the column with most frequent value 
data_df['Embarked'] = data_df['Embarked'].fillna(freq_port)

#print data_df.Embarked.isnull().values.sum() #Confirms that all null colmns are filled
#Analysing the filled column
#print data_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by= 'Survived', ascending= 'False')

#Completemissing entries of Age on the basis of Sex and Pclass
#finding how many entries of Age are null

#print data_df.Age.isnull().values.sum() #263 null entries

#Guessing ages for each Sex and Passenger
guess_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
			guess_df = data_df[(data_df['Sex'] == i) & (data_df['Pclass'] == j + 1)]['Age'].dropna()
			age_mean = guess_df.mean()
			age_std = guess_df.std()
			age_guess = np.random.normal(age_mean, age_std)

			print age_guess 
			guess_ages[i,j] = int((age_guess/0.5 + 0.5) * 0.5)