
import csv
import sys
# from itertools import groupby
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
%matplotlib inline

# Question 4(a)

data = pd.read_table("./data/u.data",  names=['user id','item id','rating', 'timestamp'])
user = pd.read_table("./data/u.user", delimiter='|',names=['user id','age','gender','occupation','zip code'])
item = pd.read_table("./data/u.item", delimiter='|', names=['item id','movie title','release date','video release date','imdb url',' unknown','action ','adventure','animation','children',' comedy','crime','documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war','western'])

movieIds = data.groupby('item id')
userIds = data.groupby('user id')
Number_of_unique_movies = len(movieIds)
Number_of_unique_users = len(userIds)
print "Number_of_unique_movies", Number_of_unique_movies, '\n', "Number_of_unique_users", Number_of_unique_users  

# Question 4(b)
moviesRating = data[['item id','rating']]
movieRatingGrouped = moviesRating.groupby('item id', as_index=False).count().sort('rating', ascending=False).head(5)
print pd.merge(movieRatingGrouped, item, how='inner', on=['item id'])[['movie title','rating']]

# Question 4(c)
mGrp = moviesRating.groupby('item id', as_index=False).count()
movie100 = mGrp[mGrp['rating']>100];
joinData = pd.merge(data, user, how='inner', on=['user id'])
of = joinData[['item id','age']]
ofGrouped = of.groupby('item id', as_index=False).mean()
ofJoin = pd.merge(movie100, ofGrouped, how='inner', on=['item id']).sort('age', ascending=True).head(5)
print pd.merge(ofJoin, item, how='inner', on=['item id'])[['movie title','rating','age']]

# Question 4(d)

spData = joinData[['rating','age']]
spf = spData.groupby('age', as_index=False).mean()
sns.jointplot(x="age", y="rating", data=spf);

# Question 4(Bonus)

g = sns.jointplot(x="age", y="rating", data=spf, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="o")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$age$", "$rating$");
