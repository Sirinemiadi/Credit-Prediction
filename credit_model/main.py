import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle



df=pd.read_csv('C:/workspace/credit_prediction_project/credit_model/train.csv')
print(df.head())
#pd.set_option('display.max_rows',df.shape[0]+1 )
#print(df.isnull().sum().sort_values(ascending=False))
#print(df.describe())

#Renseigner les valeurs manquantes
cat_data=[]
num_data=[]
for i,c in enumerate(df.dtypes):
    if c == 'object':
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()



#Pour les variables catégoriques on va remplacer les variables manquantes par la variable la plus frequentes
cat_data=cat_data.apply( lambda x : x.fillna(x.value_counts().index[0]))



#Pour les variables numeriques on va remplacer les valeurs manquantes par la valeur precedente de la meme colonne
num_data.fillna(method='bfill', inplace=True)

# Tranformer la colonne target
target_value={'Y':1,'N':0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status',axis=1,inplace=True)
target=target.map(target_value)


#encoder les variables categoriques
le = LabelEncoder()
for i in cat_data : 
   cat_data[i] = le.fit_transform(cat_data[i])

#supprimer load_ID
cat_data.drop('Loan_ID',axis=1,inplace=True)

#concatener cat_data et num_data
X=pd.concat([cat_data,num_data],axis=1)
y=target

#visualisation des données de target

'''plt.figure(figsize=(8, 6))
sns.countplot(x=target)
plt.show()

# la base de données utilisée pour EDA
df=pd.concat([cat_data,num_data,target],axis=1)

# visualisation des données de Credit history 
grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History')
plt.show()

# visualisation des données de sexe
grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender')
plt.show()

# visualisation des données des mariés 
grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'Married')
plt.show()

# visualisation des données d' education
grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education')
plt.show()


# Diviser la base de données en une base de données test et d'entrainement
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X,y):
  X_train,X_test=X.iloc[train],X.iloc[test]
  y_train,y_test=y.iloc[train],y.iloc[test]

print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)




# On va appliquer tois algorithmes Logisitic Regression, KNN, DecisionTree
models={
    'LogisticRegression':LogisticRegression(random_state=42),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1,random_state=42)
}

# La fonction de précision
def accu(y_true,y_pred,retu=False):
  acc=accuracy_score(y_true,y_pred)
  if retu:
    return acc
  else:
    print(f'la precision du modèle est: {acc}')

#c'est la fonction d'application des modèles
def train_test_eval(models,X_train,y_train,X_test,y_test):
  for name,model in models.items():
    print(name,':')
    model.fit(X_train,y_train)
    accu(y_test,model.predict(X_test))
    print('-'*30)

train_test_eval(models,X_train,y_train,X_test,y_test)'''

X_2=X[['Credit_History','Married','CoapplicantIncome']]
# Diviser la base de données en une base de données test et d'entrainement
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X_2,y):
  X_train,X_test=X_2.iloc[train],X_2.iloc[test]
  y_train,y_test=y.iloc[train],y.iloc[test]
# Appliquer la regression logisitique sur notre base de donnée
Classifier=LogisticRegression()
Classifier.fit(X_2,y)

# Enregistrer le modèle
pickle.dump(Classifier,open('model.pkl','wb'))