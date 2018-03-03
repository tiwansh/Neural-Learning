import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#1 -----------------------Take a look at data----------------------
#print train_df.info()
#print train_df.isnull().sum()ld

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

#C--------------ompletemissing entries of Age on the basis of Sex and Pclass---------#
#finding how many entries of Age are null

#print data_df.Age.isnull().values.sum() #263 null entries

#Guessing ages for each Sex and Passenger
guess_ages = np.zeros((2,3))

#Printing null ages prior to transform
#print data_df.Age.isnull().sum()

for i in range(0,2):
	for j in range(0,3):
			guess_df = data_df[(data_df['Sex'] == i) & (data_df['Pclass'] == j + 1)]['Age'].dropna()
			age_mean = guess_df.mean()
			age_std = guess_df.std()
			age_guess = np.random.normal(age_mean, age_std)

			#print age_guess 
			guess_ages[i,j] = int((age_guess/0.5 + 0.5) * 0.5)

for i in range(0,2):
	for j in range(0, 3):
			data_df.loc[(data_df.Age.isnull()) & (data_df.Sex == i) & 
			(data_df.Pclass == j+1), 'Age'] = guess_ages[i,j]


data_df['Age'] = data_df['Age'].astype(int)

#Printing number of null ages after the transform 
#print data_df.Age.isnull().sum()

#Visualising that how data is very scattered
#grid = sns.FacetGrid(data_df, col = 'Survived')
#grid.map(plt.scatter,'Age','PassengerId')
#plt.show()


#Dividing the data into 5 different data bands
data_df['AgeBand'] = pd.qcut(data_df['Age'],5)

#print data_df[['AgeBand','Survived']].groupby(['AgeBand'],as_index= False).mean().sort_values(by='AgeBand',ascending=True)

#Now based on the limits shown in the previous output, create 5 different features 
#print data_df

data_df.loc[data_df['Age'] <= 16 , 'AgeBin'] = 0
data_df.loc[(data_df['Age'] > 16) & (data_df['Age'] <= 32), 'AgeBin'] = 1
data_df.loc[(data_df['Age'] > 32) & (data_df['Age'] <= 48), 'AgeBin'] = 2
data_df.loc[(data_df['Age'] > 48) & (data_df['Age'] <= 64), 'AgeBin'] = 3
data_df.loc[(data_df['Age'] > 64), 'AgeBin'] = 4

#Drop the AgeBand once the AgeBin is populated since it will no longer be used.#
data_df = data_df.drop(['AgeBand'], axis=1)

###########Special attention to be given to AgeBin and AgeBand. THEY ARE DIFFERENT and AGEBAND is dropped
###########once AgeBin is successfully created.  

#print data_df.AgeBin.isnull().values.sum()

#checkig how many empty values we still have in the new feature 

#---------------complete the fare feature-------------------#
#print data_df.Fare.isnull().values.sum() #checking for null values 
data_df['Fare'].fillna(data_df['Fare'].dropna().median(),inplace = True)
#print data_df.Fare.isnull().values.sum() #checking for null values


data_df['FareBand'] = pd.qcut(data_df['Fare'], 5)

#Groupwise printing of the FareBand and their respective survival rates 
#print data_df[['Survived','FareBand']].groupby(['FareBand'], as_index = False).mean().sort_values(by = 'FareBand')

#Defining FareBins on the basis of FareBands
data_df.loc[data_df['Fare'] <= 7.9, 'FareBin'] = 0
data_df.loc[(data_df['Fare'] > 7.9) & (data_df['Fare'] <= 10.5), 'FareBin'] = 1
data_df.loc[(data_df['Fare'] > 10.5) & (data_df['Fare'] <= 21.558), 'FareBin'] = 2
data_df.loc[(data_df['Fare'] > 21.558) & (data_df['Fare'] <= 41.579), 'FareBin'] = 3
data_df.loc[data_df['Fare'] > 41.579, 'FareBin'] = 4

#Declare the farebin as int
data_df['FareBin'] = data_df['FareBin'].astype(int)

#Drop FareBand once the AgeBand has been populated
data_df = data_df.drop(['FareBand'], axis = 1)

#print data_df

#Creating new features#
#Finding title as a regex => Multiple characters(using +) preceeded by a '.'
data_df['Title'] = data_df.Name.str.extract('([A-Za-z]+)\.', expand = False)

#Cross tabulating the Survival rate according to the Title
#print pd.crosstab(data_df['Survived'],data_df['Title'])

#Reducing the number of titles

data_df['Title'] = data_df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Lady', 'Dona'], 'Rare')

data_df['Title'] = data_df['Title'].replace(['Mlle'], 'Miss')
data_df['Title'] = data_df['Title'].replace(['Ms'], 'Miss')
data_df['Title'] = data_df['Title'].replace(['Mme'], 'Mrs')

#Checking the available titles after modification
#print data_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean().sort_values(by = 'Title')

#print data_df.Title.isnull().values.sum()

#Create a new feature Fmimly size#
data_df['FamilySize'] = data_df['Parch']+ data_df['SibSp'] + 1

#Checking the survival on the basis of FamilySize
#grid = sns.FacetGrid(data_df, col = 'Survived')
#grid.map(plt.hist, 'FamilySize')
#plt.show()

#Creating a new feature isAlone
data_df['IsAlone'] = 0
data_df.loc[(data_df['FamilySize'] == 1 ), 'IsAlone'] = 1
#print data_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean()

#Creating a new feature AgeBin * PClass

data_df['AgeBin*Pclass'] = data_df.AgeBin * data_df.Pclass
#print data_df.Pclass.unique()
#print data_df.AgeBin.unique()
#print data_df[['AgeBin*Pclass', 'Survived']].groupby(['AgeBin*Pclass'], as_index=False).mean()

#Create a new feature - name of the lengh
data_df['NameLength'] = (data_df['Name'].apply(len))
#print data_df

#Create a new feature -> HasCabin
data_df['HasCabin'] = data_df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
#print data_df[['HasCabin', 'Survived']].groupby(['HasCabin'], as_index=False).mean()


#--------------------------------FEATURE SELECTION----------------------------#
drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin','AgeBin', 'AgeBin*Pclass','FareBin']
data_df = data_df.drop(drop_features, axis = 1)
print data_df