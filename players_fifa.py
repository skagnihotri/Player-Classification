# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:38:37 2019

@author: Shubham
"""
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
dataset = pd.read_csv('data.csv')
dataset.drop([i for i in range(13236,13284)], inplace=True)
df = dataset.loc[:,['Age', 'Overall', 'Potential', 'Special', 'International Reputation',
       'Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']].copy()
corr = df.corr()
sns.pairplot(df);
plt.show()

sns.heatmap(df.corr())
plt.show()

X = df.values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

#clusturing
from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram = dendrogram(linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('players')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=8)
y_hc = hc.fit_predict(X)

dataset['player_category'] = y_hc
dataset.to_csv('final_fifa.csv', index = False)

