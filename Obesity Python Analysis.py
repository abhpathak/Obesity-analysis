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

import json



def extrapolate_2010(col1,col2,year1,year2,pop):
    val_2010=(col1+(col2-col1)*(2010-year1)/(year2-year1)).round(0)
    return val_2010*1000/pop


os.chdir('C:\Carlson\Python\Project')

#mapping=pd.read_excel("Mapping.xlsx")

xls = pd.ExcelFile('DataDownload.xls')

main=xls.parse('Supplemental Data - County')
main.columns=main.columns.str.strip()
main=main.rename(columns={"FIPS Code": "FIPS"})
df_base=main[['FIPS','County Name','State']]

cols = pd.read_excel('Columns.xlsx')


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
    
df_base['PCT_OBESE10'] = ((df_base['PCT_18YOUNGER10']*df_base['PCT_OBESE_CHILD11']) + ((100-df_base['PCT_18YOUNGER10'])* df_base['PCT_OBESE_ADULTS10']))/100
#df_base.isnull().sum() 
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

cols=['FFRPTH10','FSRPTH10','RECFACPTH10','PCT_NHWHITE10','PCT_NHBLACK10','PCT_HISP10', \
'PCT_NHASIAN10','PCT_65OLDER10','PCT_18YOUNGER10','POVRATE10','MEDHHINC10','SNAPSPTH10','WICSPTH10','METRO13','PCT_LACCESS_HHNV10', \
'PCT_LACCESS_POP10','GROCPTH10','SUPERPTHC10','CONVSPTH10','SPECSPTH10','PC_SNAPBEN10']



col_num=df_main.ix[:,'2010 Census Population':].columns
df_norm=df_main.copy()
df_norm[col_num] = df_norm[col_num].apply(lambda x: (x - x.mean()) / (x.std()))
df_norm['METRO13']=df_main['METRO13']


cols1='FFRPTH10+FSRPTH10+RECFACPTH10+PCT_NHWHITE10+PCT_NHBLACK10+PCT_HISP10+ \
PCT_NHASIAN10+PCT_65OLDER10+PCT_18YOUNGER10+POVRATE10+MEDHHINC10+METRO13+GROCPTH10+SUPERPTHC10+CONVSPTH10+SPECSPTH10+SNAPSPTH10'

cols1='FSRPTH10+RECFACPTH10+PCT_NHWHITE10+PCT_NHBLACK10+PCT_HISP10+ \
PCT_NHASIAN10+PCT_18YOUNGER10+SNAPSPTH10+MEDHHINC10'


result = sm.ols(formula='PCT_OBESE10 ~ ' + cols1, data=df_norm).fit()
print result.summary()

result_use = sm.ols(formula='PCT_OBESE10 ~ ' + cols1, data=df_main).fit()
print result_use.summary()


#residual plot
#fig,ax = plt.subplots(figsize=(12,8))
#fig = smg.graphics.plot_ccpr(result,"FSRPTH10", ax=ax)

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
plot_columns=pca_2.fit_transform(df_final)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=model.labels_)
plt.show()


model.cluster_centers_

final = pd.DataFrame(np.nan, index=cols, columns=[1,2,3,4])

final[1] = model.cluster_centers_[0]
final[2] = model.cluster_centers_[1]
final[3] = model.cluster_centers_[2]
final[4] = model.cluster_centers_[3]
print len(DataFrame(model.labels_, columns=['Label']))
print DataFrame(model.labels_, columns=['Label']).groupby('Label').size()
print final

df_append['cluster_num']=model.labels_
df_append=DataFrame(df_append['cluster_num'],columns=['cluster_num'])


df_clus = pd.merge(df_base, df_append, left_index=True, right_index=True, how='inner');

mean_obesity=df_clus.groupby(['cluster_num'])['PCT_OBESE10'].mean().sort_values()

#dict.fromkeys(mean_obesity.index)
#clus_order={mean_obesity.index.values}


clus_dict=dict(zip(mean_obesity.index,[x for x in range(0,len(mean_obesity.index))]))
df_clus['cluster']=df_clus['cluster_num'].apply(lambda row: clus_dict[row])



#def clus_name(clus_num):
#    clus_map = {0: "a", 1: "b", 2: "c",3: "d"}
#    return clus_map[clus_num]
#
#df_clus['clus_profile']=df_clus['cluster'].apply(lambda row: clus_name(row))



with open('us_counties.topo.json') as json_data:
    d = json.load(json_data)

county_list=[]
for i in range(0,len(d['objects']['us_counties.geo']['geometries'])):
    county_list.append(d['objects']['us_counties.geo']['geometries'][i]['properties']['FIPS'])

county_list=Series(county_list)


county_list=DataFrame(county_list,columns=["FIPS"])
df_counties=pd.merge(county_list, df_clus, left_on="FIPS", right_on="FIPS",how='left')
df_counties['cluster']=df_counties['cluster'].fillna(-1)
df_counties['cluster']=df_counties['cluster']+1

os.chdir('C:\Users\Vaishnavi\Documents\Notebooks')

df_output= df_counties[['FIPS','cluster']]
df_output=df_output.sort_values(by='FIPS')
df_output.to_csv('Cluster_counties.csv', sep=',')






