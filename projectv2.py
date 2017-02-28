# -*- coding: utf-8 -*-
import sklearn
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
from scipy import stats
from pandas.stats.api import ols
import statsmodels.formula.api as sm
import statsmodels.api as smg
import statsmodels.stats.outliers_influence as stm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as sl
from sklearn.decomposition import PCA
from pylab import *
import seaborn as sns
import json



def extrapolate_2010(col1,col2,year1,year2,pop):
    val_2010=(col1+(col2-col1)*(2010-year1)/(year2-year1)).round(0)
    return val_2010*1000/pop


os.chdir('C:\Carlson\Python\Project')

xls = pd.ExcelFile('DataDownload.xls')

main=xls.parse('Supplemental Data - County')
main.columns=main.columns.str.strip()
main=main.rename(columns={"FIPS Code": "FIPS"})
df_base=main[['FIPS','County Name','State']]

cols = pd.read_excel('Columns.xlsx')
states=pd.read_csv('States.csv')

for sheet in (cols['Sheet']).unique():
    sheet_df=cols[cols['Sheet']==sheet]
    col_name=list(sheet_df['Column Name'])
    col_name.append('FIPS')
    sheet_req=xls.parse(sheet,na_values='<Null>')
    sheet_req.columns=sheet_req.columns.str.strip()
    sheet_req=sheet_req.rename(columns={"FIPS Code": "FIPS"})
    req_df=sheet_req[col_name]
    req_df=req_df.drop_duplicates()
    df_base=pd.merge(df_base,req_df,how='left')
 
    
df_base=pd.merge(df_base, states, left_on="State", right_on="Abbrev",how='left')
colorder=df_base.columns.values

#df_base.isnull().sum() 
df_base['PCT_OBESE10'] = ((df_base['PCT_18YOUNGER10']*df_base['PCT_OBESE_CHILD11']) + ((100-df_base['PCT_18YOUNGER10'])* df_base['PCT_OBESE_ADULTS10']))/100
df_base['OBESE10']=df_base['PCT_OBESE10']*df_base['2010 Census Population']/100
df_base['DIABETES10']=((100-df_base['PCT_18YOUNGER10'])* df_base['PCT_DIABETES_ADULTS10'])*df_base['2010 Census Population']/10000
df_base['RECFACPTH10']=extrapolate_2010(df_base['RECFAC07'],df_base['RECFAC12'],2007,2012,df_base['2010 Census Population'])
df_base['FFRPTH10']=extrapolate_2010(df_base['FFR07'],df_base['FFR12'],2007,2012,df_base['2010 Census Population'])
df_base['FSRPTH10']=extrapolate_2010(df_base['FSR07'],df_base['FSR12'],2007,2012,df_base['2010 Census Population'])
df_base['SNAPSPTH10']=extrapolate_2010(df_base['SNAPS08'],df_base['SNAPS12'],2008,2012,df_base['2010 Census Population'])
df_base['WICSPTH10']=extrapolate_2010(df_base['WICS08'],df_base['WICS12'],2008,2012,df_base['2010 Census Population'])
df_base['GROCPTH10']=extrapolate_2010(df_base['GROC07'],df_base['GROC12'],2007,2012,df_base['2010 Census Population'])
df_base['SUPERPTHC10']=extrapolate_2010(df_base['SUPERC07'],df_base['SUPERC12'],2007,2012,df_base['2010 Census Population'])
df_base['CONVSPTH10']=extrapolate_2010(df_base['CONVS07'],df_base['CONVS12'],2007,2012,df_base['2010 Census Population'])
df_base['SPECSPTH10']=extrapolate_2010(df_base['SPECS07'],df_base['SPECS12'],2007,2012,df_base['2010 Census Population'])

df_main=df_base.dropna(subset=['PCT_OBESE_ADULTS10','PCT_OBESE_CHILD11'])



df_state=df_main[['State Name','OBESE10','DIABETES10','2010 Census Population','State']].groupby(['State Name','State'], as_index=False).sum()
df_state['PctObeseState'] = df_state['OBESE10']/df_state['2010 Census Population']*100
df_state['PctDiabetesState'] = df_state['DIABETES10']/df_state['2010 Census Population']*100
print 'Top 5 Obese States:'
print df_state.sort_values('PctObeseState', ascending = False)['State Name'][0:5]
print 'Top 5 Diabetic States:'
print df_state.sort_values('PctDiabetesState', ascending = False)['State Name'][0:5]


print '\nLeast 5 Obese States:'
print df_state.sort_values('PctObeseState')['State Name'][0:5]
print '\nLeast 5 Diabetic States:'
print df_state.sort_values('PctDiabetesState')['State Name'][0:5]

labels = df_state['State']
colors = np.random.rand(len(df_state))
fig = plt.figure(facecolor='white')
plt.scatter(df_state['PctObeseState'], df_state['PctDiabetesState'],s=df_state['2010 Census Population']*1000/max(df_state['2010 Census Population']),c=colors,alpha=0.5)

for label, x, y in zip(labels, df_state['PctObeseState'], df_state['PctDiabetesState']):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.25', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.ylabel('Diabetes Percentage',fontsize=25)
plt.xlabel('Obesity Percentage',fontsize=25)
plt.title('Diabetes vs Obesity for all states',fontsize=30)
plt.tick_params(axis='both', labelsize=20)
plt.show()

def metro_name(clus_num):
    clus_map = {0: "Metro", 1: "Non-metro"}
    return clus_map[clus_num]

df_main['Metro_name']=df_main['METRO13'].apply(lambda row: metro_name(row))

t, p = stats.ttest_ind(df_main[df_main['METRO13']==0]['PCT_OBESE_ADULTS10'].values.astype(int), df_main[df_main['METRO13']==1]['PCT_OBESE_ADULTS10'].values.astype(int), equal_var=False)
print "For the two sample t-test, t-statistic is {:.3f} and the p-value is {:.3f}" .format(t, p)

ax=sns.boxplot(x="Metro_name", y="PCT_OBESE_ADULTS10",  data=df_main)
ax.set_xlabel(None,fontsize=0)
ax.set_ylabel("Obesity Percentage",fontsize=25)
ax.set_title("Obesity across metro/non-metro cities",fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=20)

plt.show()

df_main = df_main.drop('State Name', 1)
df_main = df_main.drop('Abbrev', 1)
df_main = df_main.drop('Metro_name', 1)

col_num=df_main.ix[:,'2010 Census Population':].columns
df_norm=df_main.copy()
df_norm[col_num] = df_norm[col_num].apply(lambda x: (x - x.mean()) / (x.std()))
df_norm['METRO13']=df_main['METRO13']

cols=['FFRPTH10','FSRPTH10','RECFACPTH10','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10', \
'PCT_NHASIAN10','PCT_65OLDER10','PCT_18YOUNGER10','POVRATE10','MEDHHINC10','METRO13','SNAPSPTH10','GROCPTH10','SUPERPTHC10','CONVSPTH10','SPECSPTH10','WICSPTH10']

df_corr=df_main[cols].apply(lambda x: x.corr(df_main['PCT_OBESE_ADULTS10']))
df_corr.sort_values()

cols1='FFRPTH10+FSRPTH10+RECFACPTH10+PCT_NHWHITE10+PCT_NHBLACK10+PCT_HISP10+ \
PCT_NHASIAN10+PCT_65OLDER10+PCT_18YOUNGER10+POVRATE10+MEDHHINC10+METRO13+SNAPSPTH10+GROCPTH10+SUPERPTHC10+CONVSPTH10+SPECSPTH10+WICSPTH10'

result = sm.ols(formula='PCT_OBESE10 ~ ' + cols1, data=df_norm).fit()
print result.summary()

cols1='FSRPTH10+RECFACPTH10+PCT_NHWHITE10+PCT_NHBLACK10+PCT_HISP10+ \
PCT_NHASIAN10+MEDHHINC10+METRO13+SNAPSPTH10+GROCPTH10+SUPERPTHC10'

result = sm.ols(formula='PCT_OBESE10 ~ ' + cols1, data=df_norm).fit()
print result.summary()


result_use = sm.ols(formula='PCT_OBESE10 ~ ' + cols1, data=df_main).fit()
print result_use.summary()

figure-smg.graphics.plot_ccpr(result_use,'FSRPTH10')
figure.show()


plt.plot(df_main['PCT_OBESE10'], result_use.predict(df_main[cols]), color='blue',
         linestyle='dotted')
plt.show()
#Predict missing obese values
df_missing=df_base[~df_base.index.isin(df_main.index)]

cols=['FSRPTH10','RECFACPTH10','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10', \
'PCT_NHASIAN10','PCT_18YOUNGER10','MEDHHINC10','SNAPSPTH10']

df_pred=df_missing[cols]
df_pred=df_pred.dropna(how='any')
df_pred['PCT_OBESE10']=result_use.predict(df_pred)

cols=['PCT_OBESE10','RECFACPTH10','FSRPTH10','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10', \
'PCT_NHASIAN10','PCT_18YOUNGER10','MEDHHINC10','SNAPSPTH10']

df_append=pd.concat([df_main[cols],df_pred[cols]])

df_final=df_append[cols]
model = KMeans(n_clusters=4)
model.fit(df_final)

print model.fit

pca_2=PCA(2)
plt.figure(3)
plot_columns=pca_2.fit_transform(df_final)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=model.labels_)
plt.show()


model.cluster_centers_

df_append['cluster_num']=model.labels_
df_append=DataFrame(df_append['cluster_num'],columns=['cluster_num'])


df_clus = pd.merge(df_base, df_append, left_index=True, right_index=True, how='inner');

mean_obesity=df_clus.groupby(['cluster_num'])['PCT_OBESE10'].mean().sort_values()


clus_dict=dict(zip(mean_obesity.index,[x for x in range(0,len(mean_obesity.index))]))
df_clus['cluster']=df_clus['cluster_num'].apply(lambda row: clus_dict[row])


with open('us_counties.topo.json') as json_data:
    d = json.load(json_data)

county_list=[]
for i in range(0,len(d['objects']['us_counties.geo']['geometries'])):
    county_list.append(d['objects']['us_counties.geo']['geometries'][i]['properties']['FIPS'])

county_list=Series(county_list).unique()


county_list=DataFrame(county_list,columns=["FIPS"])
df_counties=pd.merge(county_list, df_clus, left_on="FIPS", right_on="FIPS",how='left')
df_counties['cluster']=df_counties['cluster'].fillna(-1)
df_counties['cluster']=df_counties['cluster']+1

os.chdir('C:\Users\Vaishnavi\Documents\Notebooks')

df_output= df_counties[['FIPS','cluster','State']]
df_output=df_output.sort_values(by='FIPS')
df_output.to_csv('Cluster_counties.csv', sep=',')




cluster_pct=((df_clus.groupby('cluster').size()/df_clus.groupby('cluster').size().sum())*100)
cluster_pct=cluster_pct.round(0).astype(int)


#Pie Chart

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Physically Fit', 'Moderate', 'Aboriginals', 'Obese'
fracs = [15, 30, 45, 10]
explode=(0, 0, 0, 0.05)

plt.pie(cluster_pct, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90,colors=['green', 'lightgreen', 'blue', 'red'])
  
title('Cluster distribution', bbox={'facecolor':'0.8', 'pad':5})
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show()  








