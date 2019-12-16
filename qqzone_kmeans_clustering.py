from pytoolkit import TDWSQLProvider,TableDesc,TableInfo,TDWUtil
from pyspark.sql import SQLContext,Row
from pyspark.sql.functions import lit,md5,concat,col,count,broadcast,variance,bround
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer,StringIndexerModel

import xgboost as xgb
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_boston

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import sparse,stats
import pandas as pd
from operator import attrgetter
from sys import getsizeof
from math import sqrt
import datetime
import copy
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
from sklearn.decomposition import PCA

sc._conf.set("spark.tdw.orcfile.datasource","false")
sc._conf.set("spark.driver.maxResultSize","12g")
print(sc.applicationId)
print sc._conf.get('spark.driver.memory')
print sc._conf.get('spark.driver.maxResultSize')


# 定义元数据信息
meta_group, db_name, tb_name = '同乐', 'u_isd_qzone', 'qqzone_user_clustering'
# 初始化TDWProvider，无需显示指定用户名和密码
tdw = TDWSQLProvider(spark, db=db_name, group=meta_group)
# # 提取表数据
# mydf = tdw.table(tb_name).select('ftime', 'tag').limit(10).collect()
# # 输出结果
# mydf.show()

# `sqlCtx`是内置的当前spark.sqlContext的引用，可以直接使用
sqlCtx.registerDataFrameAsTable(tdw.table(tb_name), 'tb_name')
# 使用SQL语句查询数据
# mydf = sqlCtx.sql('select * from tb_name where ftime=20191101')#.collect()
mydf = sqlCtx.sql('select * from tb_name').toPandas()#.collect()


#缺失数据用-1填补
mydf.loc[mydf['age'].isnull(),'age']=mydf.loc[mydf['age'].isnull(),'age'].fillna(-1)

def detect_outliers(df,n,features): #筛选异常值
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices=[]
    for col in features:
        Q1=np.percentile(df[col], 25)
        Q3=np.percentile(df[col],75)
        outlier_step=3*(Q3 - Q1)
        outlier_list_col=df[df[col]>Q3+outlier_step].index
        outlier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices=Counter(outlier_indices)        
    multiple_outliers=list(k for k,v in outlier_indices.items() if v>n)
    return multiple_outliers   

Outliers_to_drop=detect_outliers(mydf,4,mydf.columns.tolist()[2:]) #异常值处理
mydf=mydf.drop(Outliers_to_drop,axis=0).reset_index(drop=True)

def data_desc(df_table):  #数据集描述
    t = []
    n = df_table.shape[0]
    for vals in df_table.columns.values.tolist():
        n_dtype = df_table[vals].dtype  #变量类型
        #         n=df_table[vals].count() #变量个数
        n_distinct = df_table[vals].drop_duplicates().count()  #变量去重个数
        n_miss = sum(pd.isnull(df_table[vals]))  #缺失值个数
        pct_miss = np.round(n_miss * 100 / n, 2)  #缺失比例

        if df_table[vals].dtype != 'object':
            n_min = df_table[vals].min()  #最小值
            n_max = df_table[vals].max()  #最大值
            n_mean = df_table[vals].mean()
            n_std = df_table[vals].std()
            q = stats.scoreatpercentile(df_table[vals].dropna(),
                                        [5, 25, 50, 75, 99.9])  #分位数
        else:
            n_min = np.nan
            n_max = np.nan
            n_mean = np.nan
            n_std = np.nan
            q = [np.nan, np.nan, np.nan, np.nan, np.nan]
        q_low = q[1] - 3 * (q[3] - q[1])  #75分位数-25分位数（异常值下界）
        q_upper = q[3] + 3 * (q[3] - q[1])  #75分位数-25分位数（异常值上界）
        t.append(
            (vals, n_dtype, n, n_distinct, n_miss, pct_miss, n_min, n_max,
             n_mean,n_std,q[0], q[1], q[2], q[3], q[4],
             (df_table[vals] < q_low).sum() + (df_table[vals] > q_upper).sum(),
             df_table.loc[(df_table[vals] > q_upper) |
                          (df_table[vals] < q_low)].index.tolist()[0:3]))
    colnums = [
        'colums', 'n_dtype', 'n_num', 'n_distinct', 'n_miss', 'pct_miss',
        'n_min', 'n_max', 'n_mean','n_std', 'q_5', 'q_25', 'q_50', 'q_75', 'q_99',
        'n_outliers', 'outliers_index'
    ]
    df_table1 = pd.DataFrame(t, columns=colnums)
    return df_table1
	
#将超过99.9分位的原始数据人工修改为99分位
for var_name in mydf.columns.tolist()[1:]:
    p_99=np.percentile(mydf[var_name], 99.9)
    mydf[var_name]=mydf[var_name].apply(lambda x:p_99 if x>p_99 else x)
	
#相关性检验
plt.figure(figsize=(10, 8), dpi=80, edgecolor='k').add_subplot(1,1,1)
sns.heatmap(mydf.corr(),annot=True,cbar=True,cmap='RdGy',fmt = ".2f",center=0)
plt.show()

#计算tf_idf值，对热门功能进行惩戒
def tf_idf(var):
    tf=mydf[var]/mydf['all_write_cnt']
    idf=np.log(mydf.shape[0]/mydf[mydf[var]>0].shape[0])
    return tf*idf

data_tmp2=copy.deepcopy(mydf)
for var in data_tmp2.columns.tolist()[3:]:
    data_tmp2['pct_%s'%var]=tf_idf(var)
    data_tmp2.drop('%s'%var,inplace=True,axis=1) #删除部分变量
data_tmp2.drop('all_write_cnt',inplace=True,axis=1) #删除部分变量

#部分变量存在幂指分布
fig = plt.figure(figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
sns.set_style('darkgrid')
sns.kdeplot(mydf['shuoshuo_cnt'],color='black',shade=True,)
plt.show()

#对部分幂指分布的用户进行log线性处理
data_tmp1=copy.deepcopy(mydf)
for var in data_tmp1.columns.tolist():
    data_tmp1['%s_log'%var]=data_tmp1['%s'%var].map(lambda i:np.log10(i+2)) 
    data_tmp1.drop('%s'%var,inplace=True,axis=1) #删除部分变量
	
#数据标准化
data_tmp2=pd.DataFrame(preprocessing.scale(data_tmp1),columns=data_tmp1.columns)

#PCA降维
pca=PCA(n_components=3)
data_tmp3=pca.fit_transform(data_tmp2)

datMat=np.asmatrix(data_tmp3)



def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 计算欧式距离


def randCent(dataSet, k): #随机选择k个质心
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k,n))) #创建0矩阵
    for j in range(n): #创建随机质心
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #寻找最近的质心
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
#         print(centroids,len(centroids))
        for cent in range(k): #更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#存储聚类结果
            centroids[cent,:] = mean(ptsInClust, axis=0)  
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #创建初始类
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)): #划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] 
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])# 比较SSE的大小
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
#             print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
#         print('the bestCentToSplit is: ',bestCentToSplit)
#         print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #更新簇的分配结果
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#存储聚类结果及SSE值
    return mat(centList), clusterAssment
	
	
clustering_sse=[]
for k in np.linspace(2,20,19).astype(int):
    myCentroids,clustAssing=biKmeans(datMat,k)
    clustering_sse.append(sum(clustAssing[:,1]))

#通过SSE选择最佳K的取值
sns.set(style="darkgrid", palette="muted", color_codes=True) 
# sns.set( palette="muted", color_codes=True)  
plt.figure(figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
sns.pointplot(np.linspace(2,20,19).astype(int),clustering_sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()

#通过轮廓系数判断聚类效果
distortions=[]
for i in np.linspace(2,10,9).astype(int):
    y_pred=KMeans(n_clusters=i,n_jobs = 4).fit_predict(data_tmp3)
    distortions.append(silhouette_score(data_tmp3, y_pred))
plt.plot(np.linspace(2,10,9).astype(int),distortions)  

myCentroids,clustAssing=biKmeans(datMat,8)

data_tmp2['y_pred']=map(int,clustAssing[:,0].A)


#绘制聚类雷达图
data_tmp1_gb=data_tmp2.groupby(['y_pred']).mean().reset_index()
data_tmp1_gb_nm=data_desc(data_tmp1_gb.iloc[:,1:]).sort_values(['n_std'],ascending=False).iloc[0:8,:]['colums'].tolist()
data_tmp1_gb=data_tmp1_gb[['y_pred']+data_tmp1_gb_nm]

values_tmp1,angles_tmp1,cluster_result=[],[],np.unique(data_tmp2['y_pred'])
for i in cluster_result:
    feature=data_tmp1_gb.columns.tolist()[1:]
    values = np.array(data_tmp1_gb[data_tmp1_gb['y_pred']==i].iloc[:,1:]).tolist()[0]
    angles=np.linspace(0, 2*np.pi, len(values), endpoint=False)
    values_tmp1.append(np.concatenate((values,[values[0]])))
    angles_tmp1.append(np.concatenate((angles,[angles[0]])))

ax=plt.figure(figsize=(10, 8), dpi=80, edgecolor='k').add_subplot(111,polar=True)
for i in cluster_result:
    ax.plot(angles_tmp1[i], values_tmp1[i], 'o-', linewidth=2, label = i) # 绘制折线图
    ax.fill(angles_tmp1[i], values_tmp1[i], alpha=0.25) # 填充颜色
    ax.set_thetagrids(angles_tmp1[i] * 180/np.pi, feature) # 添加每个特征的标签
ax.set_ylim(-2,5) # 设置雷达图的范围
ax.grid(True) # 添加网格线
plt.legend(loc = 'best') # 设置图例
plt.show() # 显示图形

#绘制聚类结果图
L1=[n[0] for n in data_tmp3]
L2=[n[1] for n in data_tmp3]
plt.rcParams['axes.unicode_minus']=False 
plt.rc('font', family='SimHei', size=8)
fig = plt.figure(figsize=(20,20), dpi=80, facecolor='w', edgecolor='k')
p1 = plt.subplot(221)
plt.title(u"Kmeans聚类")
plt.scatter(L1,L2,c=data_tmp2['y_pred'],marker="s")
plt.sca(p1)
plt.show()
