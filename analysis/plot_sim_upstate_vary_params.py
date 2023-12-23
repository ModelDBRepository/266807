#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import pickle
with open('Simulations/results_wider_CaL12CaL13CaT32CaT33.pickle','rb') as f:
    d = pickle.load(f)

mod_list = d['mod_list']
results = d['results']

df = pd.DataFrame(mod_list)
last_stim_time = .108
    
amp_list = []
dur_list = []
for r in results:
    p2p = r[:,1].ptp()
    time_above_half = r[:,0][(r[:,1]-r[:,1].min())>(p2p/2.)] 
    duration = time_above_half[-1] - last_stim_time
    amp_list.append(p2p)
    dur_list.append(duration)

df = pd.concat([df,
pd.Series(amp_list,name='amp'),
pd.Series(dur_list,name='dur')],
axis=1)    

#import seaborn as sns 
#sns.pairplot(df)
#%%
f,ax = plt.subplots(1,1)
cm = plt.cm.viridis(np.linspace(0,1,len(results)))
    

#for i,r in enumerate(results):
#    ax.plot(results[i][:,0],results[i][:,1],label=str(i),color=cm[i])

#ax.legend()
plt.show()

for v in df.columns:
    if v not in ['amp','dur']:
        f,a = plt.subplots(1,2)
        a[0].scatter(df[v],df['amp'])
        
        a[0].set_xlim(df[v].min(),df[v].max())
        a[1].scatter(df[v],df['dur'])
        a[1].set_xlim(df[v].min(),df[v].max())
        
        f.suptitle(v)

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.svm import SVR

df_above50 = df[df['dur']>50e-3]
X = df_above50.drop(['amp','dur'], axis=1)

y = df_above50['dur']
X = X/X.max()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create linear regression object

#regr = linear_model.LinearRegression()
#regr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#regr = SVR(kernel='sigmoid')#, C=100, gamma='auto', degree=3, epsilon=.1,
#           # coef0=1)
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100)
#    Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#        % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.figure()
plt.scatter(X_test['nmda_gbar'], y_test,  color='black')
plt.scatter(X_test['nmda_gbar'], y_pred, color='blue')
plt.show()

plt.figure()
plt.scatter(df['dur'],df['amp'])
plt.show()


# plt.figure()
# ax = plt.subplot(111, projection='polar')

# theta = np.linspace(0, 360, X.shape[1])
# for i,c in enumerate(X.columns): 
#     t = np.zeros(len(X[c]))+theta[i]
#     r = X[c]
#     c = df['dur']
#     #c = plt.cm.viridisX['dur']
#     ax.scatter(t, r,c=c, cmap = plt.cm.viridis)
#     #ax.set_rmax(2)
#     #ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
#     #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted #line

#ax.grid(True)
#%%
df_filter_dir = df[(df['dur']>.08)&(df['dur']<1)&(df['amp']<18e-3)&(df['amp']>8e-3)].drop(['amp','dur'],axis=1)
df_filter_dir = df_filter_dir/df_filter_dir.max()
df_filter_dir = df_filter_dir[['nmda_gbar','ampa_gbar','CaLmod','CaRmod','CaTmod','KaFmod','KaSmod','Kirmod','BKCamod','SKCamod','CaCCmod']]
plt.figure()
plt.ion()
ax = plt.subplot(121, projection='polar')
ax2 = plt.subplot(122)
theta = np.linspace(0, 2*np.pi, X.shape[1]+1)
for r in df_filter_dir.iterrows(): 
    t = theta
    rc = r[1]#.drop(['amp','dur'])
    rc = list(rc)
    rc.append(rc[0])
    #c = df['dur']
    #c = plt.cm.viridisX['dur']
    ax.plot(t, rc)#,c=c, cmap = plt.cm.viridis)
    #ax.set_rmax(2)
    #ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted #line
    ax2.plot(results[r[0]][:,0],results[r[0]][:,1])

ax.set_xticks(theta)
ax.set_xticklabels(df_filter_dir.columns)
#ax.set_thetalim(-np.pi, np.pi)
ax.grid(True)

for c in df_filter_dir.columns:
    plt.figure()
    #plt.scatter(df_filter_dir['CaLmod'],df_filter_dir[c])
    plt.title(str(c))
    plt.hist(df_filter_dir[c])

rank_rmse = []
for r in df_filter_dir.iterrows(): 
    rms = np.sqrt(np.mean((r[1] - (np.zeros(len(r[1]))+.5))**2))
    #print(rms)
    rank_rmse.append((r[0],rms))


df_rmse = pd.concat([df_filter_dir, pd.DataFrame(rank_rmse,columns=['','rmse']).set_index('')],axis=1)

df_rmse.plot('rmse')

var_list = []

for r in df_rmse.sort_values('rmse').drop('rmse',axis=1).iterrows():
    vd = df.iloc[r[0]].drop(['amp','dur']).to_dict()
    vd['ID']=r[0]
    var_list.append(vd)

import pickle
with open('var_list.pickle','wb') as f:
    pickle.dump(var_list,f)

plt.figure()
bins = np.linspace(min(df['dur']),max(df['dur']),100)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(df['dur'], bins=logbins)
plt.xscale('log')
plt.show()

# TODO:
# 1. Move SK to spines
# 2. Test potential mechanism due to R-type
# 3. Can alter balance of L/R type channels in dendrite/spine for r in 
# 
