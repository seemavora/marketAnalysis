import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('rawData/modifiedDiscord.csv')
le = LabelEncoder()
conditions = ['major_department', 'year', 'music_genre', 'living',
              'humour', 'sports', 'personal_qualities', 'friend_qualities',
              'friday_night', 'school_balance', 'entertainment_genre']
for i in range(len(conditions)):
    data[conditions[i]] = le.fit_transform(data[conditions[i]])

data.to_csv('cleanData/oneEncodedDiscord.csv')
