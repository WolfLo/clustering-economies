from prep import Preprocessing

list=['Population_2014',
      'Population_2015']
dd=Preprocessing('data/selected_countries.csv',
                 varlist=None)
dataset=dd.df
ddd, pc = dd.drop_poor_columns(0.5)
pc
ddd.shape
ddd2, pc2 = dd.drop_poor_features(axis=1, p=0.5)
ddd2

pc2
axis = 0
int(not axis)
ddd2.shape
ddd.shape
