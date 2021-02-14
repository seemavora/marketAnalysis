import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 

data = pd.read_csv('rawData/PA_2.csv')
# le = LabelEncoder()
conditions =  ['major_department', 'year', 'music_genre', 'living', 
'humour', 'sports','friday_night','school_balance','entertainment_genre']
# for i in range (len(conditions)):
#   data[conditions[i]] = le.fit_transform(data[conditions[i]])
  
# onehotencoder = OneHotEncoder() 

for i in range (len(conditions)):
  columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [i])], remainder='passthrough') 
  data[conditions[i]] = np.array(columnTransformer.fit_transform(data), dtype = np.str)

# columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
# data = np.array(columnTransformer.fit_transform(data), dtype = np.str)
# one_hot = data.copy()
# one_hot = pd.get_dummies(data=data,columns=conditions)
# print(one_hot.head())

# data = data.drop('year',axis = 1)
# data = data.join(one_hot)
print(data)
# data.to_csv('cleanData/OHE_PA.csv')


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.impute import SimpleImputer
# # Importing the dataset
# dataset = pd.read_csv('rawData/personalityAnalysis.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 3].values

# # Taking care of missing data
# imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
# imputer = SimpleImputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# # Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)