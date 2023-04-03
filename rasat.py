#%% Data Analyses and Preparation Library Design
#%% Created by: Ezgi Demir, Ph.D. Operations Research,
#               Batuhan ATASOY, Ph.D. Mechtaronics Engineering

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 as sql

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm

#%% Constructing Data Class
class Data:
    def __init__(self,filename):
        self.file=filename
        self.object=self.file.replace('.sql','')
        self.sql_explanation="SELECT * from "+self.object
        self.drop_list=[]
        self.p_data_drop=[]
        self.pca_list=[]
        self.sc=StandardScaler()
        self.sinif_1=SVC(kernel='linear')
        self.sinif_2=KNeighborsClassifier(n_neighbors=7,metric='manhattan')
        self.sinif_3=RandomForestClassifier()
        self.sinif_4=XGBClassifier()

    # Accessing the SQLite Database
    def open_db(self):
        print('\n-------------Step 1: Accessing the Database-------------\n')
        try:
            self.connection=sql.connect(self.file)
            self.dataframe=pd.read_sql(self.sql_explanation, self.connection)
            self.connection.close()
            print("\n Database",self.file,"has been successfully accessed... \n")
            print("\n Data Information \n")
            print(self.dataframe.info())

        except FileNotFoundError:
            print("\n Database",self.file,"has been found... \n")

    # Cleaning the Dataframe from Irrelevant Columns
    def clean_df(self):
        print('\n-------------Step 2: Cleansing the Dataframe-------------\n')

        self.drop_feature=input('\n Enter the dropped feature(s) (with comma seperated)= ')
        self.drop_feature=self.drop_feature.split(',')

        self.new_df=self.dataframe.drop(self.drop_feature,axis=1)
        self.dim=list(self.new_df.shape)

        print("\n Dropped Features \n")
        print(self.drop_feature)
        print("\n New Data Information \n")
        print(self.new_df.info())

    # Handling Missing Values
    def handle_nan(self):
        self.stats=self.new_df.describe()
        print('\n-------------Step 3: Handling Missing Values-------------\n')

        self.le=LabelEncoder()
        self.encoded_df=[]
        
        for i in range(self.new_df.shape[1]):
            
            if self.new_df.iloc[:, i].dtype==object or self.new_df.iloc[:, i].dtype==bool:
                self.new_df.iloc[:,i]=self.new_df.iloc[:,i].fillna(value='Unknown')
                self.le_value=self.le.fit(self.new_df.iloc[:, i])
                self.le_value=self.le.transform(self.new_df.iloc[:, i])
                
            else:
                self.le_value=self.new_df.iloc[:, i]
                self.le_value=self.le_value.fillna(value=self.le_value.median())
                            
            self.encoded_df.append(self.le_value)

        self.encoded_df=pd.DataFrame(np.asarray(self.encoded_df).T,columns=self.new_df.columns)
        print(self.encoded_df.info())
        

        self.encoded_df=self.encoded_df.fillna(value=self.encoded_df.mean())
        self.stats_2=self.encoded_df.describe() 
    
    #%% Seperating the Dataframe into Dependent & Independent Pairs
    def seperate(self):
        print('\n-------------Step 4: Seperating the Dataframe into Dependent-Independent Variables-------------\n')
        
        self.y_head=input('Enter your Dependent Variable(s) (with comma seperated)=')
        self.y_head=self.y_head.split(',')
        
        self.y=self.encoded_df[self.y_head]
        self.x=self.encoded_df.drop(self.y_head,axis=1)
        
        print('\n Independent Variable(s) \n')
        print(self.x.info())
        
        print('\n Dependent Variable(s) \n')
        print(self.y.info())

    #%% Dimensional Reduction
    def reduce_dim(self):
        print('\n-------------Step 5: Feature selection and Dimensional Reduction-------------\n')
        plt.figure('Correlation Heatmap Before Feature Selection')
        plt.title('Correlation Heatmap Before Feature Selection')
        self.corr_1=self.x.corr()
        sns.heatmap(self.corr_1,annot=True,cmap=plt.cm.Reds)

        # 1. Check Multicollinearity with Correlation Matrix
        self.upper_triangle=self.corr_1.where(np.triu(np.ones(self.corr_1.shape),k=1).astype(np.bool))
        
        self.to_drop_features=[column for column in self.upper_triangle.columns if any(abs(self.upper_triangle[column])>0.5)]
        print("\n Auto-Drop Features \n",self.to_drop_features,"\n")
        
        self.filtered_df=self.x.drop(self.to_drop_features,axis=1)
        self.corr_2=self.filtered_df.corr()

        plt.figure('Correlation Heatmap After 1st Feature Selection')
        plt.title('Correlation Heatmap After 1st Feature Selection (with Correlation Matrix)')
        sns.heatmap(self.corr_2,annot=True,cmap=plt.cm.Blues)
        
        # 2. p-value Test for Dimensional Reduction
        
        self.filtered_df=sm.add_constant(self.filtered_df)
        self.first_model=sm.OLS(self.y,self.filtered_df)
        self.p_analyses=self.first_model.fit()
        print(self.p_analyses.summary())
        self.p_features=dict(self.p_analyses.pvalues)

        self.p_data_drop=[element for element in self.filtered_df if self.p_features[element]>0.05]
        
        self.filtered_df=self.filtered_df.drop(self.p_data_drop,axis=1)
        self.filtered_df=self.filtered_df.drop('const',axis=1)

        plt.figure('Correlation Heatmap After 2nd Feature Selection')
        plt.title('Correlation Heatmap After 2nd Feature Selection (with p-value Tests)')        
        sns.heatmap(self.filtered_df.corr(),annot=True,cmap=plt.cm.Greens)
        
        # 3. PCA-based Dimensional Reduction
        
        for i in range(self.filtered_df.shape[1]-1):
            self.pca=PCA(i+1,whiten=True)
            self.pca.fit(self.filtered_df)
            self.x_pca=self.pca.transform(self.filtered_df)
            self.pca_list.append(float("{:.2f}".format(100*sum(self.pca.explained_variance_ratio_))))

        # Constructing PCA for the Dataset if Variance Ratio is Greater than or Equal to 0.98
            if sum(self.pca.explained_variance_ratio_)>0.999:
                self.embedded_dim=i+1
                print("\n Number of Optimal Principal Components = {} \n".format(self.embedded_dim))
                break
        
        plt.figure("Principal Component Analyses")
        plt.plot(self.pca_list,'-r')
        plt.plot(self.pca_list,'^b')
        plt.title("Principal Component Analyses")
        plt.xlabel("The Number of Components")
        plt.ylabel("Cumulative Variance Ratio")
        plt.grid(True)

        self.pca_head=[]
        for i in range(self.embedded_dim):
            self.pca_head.append("PCA-{}".format(i+1))

        self.new_x_pca=pd.DataFrame(self.x_pca,columns=self.pca_head)

    #%% Detecting the Number of Optimal Clusters via Hierarchical and Non-Hierarchical Methods
    def cluster(self):
        print('\n-------------Step 6: Cluster Analyses to Inspect the Dataverse-------------\n')

        # Hierarchical Clustering
        self.cluster_model=AgglomerativeClustering()
        self.cluster_model=self.cluster_model.fit(self.new_x_pca)
        
        self.relation=linkage(self.cluster_model.children_)
        
        plt.figure('Dendrgoram Analyses to Detect Clusters')
        plt.title('Dendrgoram Analyses to Detect Clusters')
        self.k=dendrogram(self.relation)
        
        self.n_cluster=len(set(self.k['leaves_color_list']))
        
        print("\n Number of detected clusters={} \n".format(self.n_cluster))
        
        # Non-Hierarchical Clustering
        self.kmeans=KMeans(n_clusters=self.n_cluster)
        self.kmeans=self.kmeans.fit(self.new_x_pca)
        self.cluster_results=self.kmeans.labels_

        self.pre_results=pd.DataFrame(np.column_stack((self.y.values,self.cluster_results)),columns=['Original Data','Cluster Results'])
        
    def cv(self):
        self.skor_svc=cross_val_score(self.sinif_1,self.filtered_df,self.y, cv=10)
        self.skor_knn=cross_val_score(self.sinif_2,self.filtered_df,self.y, cv=10)
        self.skor_rfc=cross_val_score(self.sinif_3,self.filtered_df,self.y, cv=10)
        self.skor_xgbc=cross_val_score(self.sinif_4,self.filtered_df,self.y, cv=10)
        self.fold=np.linspace(1, 10, num=10)
        
        plt.figure('CV Scores')
        plt.plot(self.fold,self.skor_svc,'-^r')
        plt.plot(self.fold,self.skor_knn,'-*g')
        plt.plot(self.fold,self.skor_svc,'-+b')
        plt.plot(self.fold,self.skor_xgbc,'-<c')
        plt.legend(['SVC','KNN','Random Forest','XGBoost'])
        plt.grid(True)
        plt.title('CV Scores')
        plt.xlabel('Folds')
        plt.ylabel('Scores')
    
    def batch(self):
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x_pca,self.y,test_size=0.2,shuffle=True)
        self.x_train=self.sc.fit_transform(self.x_train)
        self.x_test=self.sc.transform(self.x_test)
        
        self.sinif_1.fit(self.x_train,self.y_train)
        self.y_svc=self.sinif_1.predict(self.x_test)
        
        self.sinif_2.fit(self.x_train,self.y_train)
        self.y_knn=self.sinif_2.predict(self.x_test)
        
        self.sinif_3.fit(self.x_train,self.y_train)
        self.y_rfc=self.sinif_3.predict(self.x_test)
        
        self.sinif_4.fit(self.x_train,self.y_train)
        self.y_xgbc=self.sinif_4.predict(self.x_test)
        
        
        plt.figure('SVC Classfier')
        plt.title('SVC Classfier')
        plt.xlabel('Classes')
        plt.ylabel('Classes')
        self.cm1=confusion_matrix(self.y_test,self.y_svc)
        sns.heatmap(self.cm1,annot=True,cmap=plt.cm.Reds)
        
        plt.figure('KNN Classfier')
        plt.title('KNN Classfier')
        plt.xlabel('Classes')
        plt.ylabel('Classes')
        self.cm2=confusion_matrix(self.y_test,self.y_knn)
        sns.heatmap(self.cm2,annot=True,cmap=plt.cm.Reds)

        plt.figure('RF Classfier')
        plt.title('RF Classfier')
        plt.xlabel('Classes')
        plt.ylabel('Classes')
        self.cm3=confusion_matrix(self.y_test,self.y_rfc)
        sns.heatmap(self.cm3,annot=True,cmap=plt.cm.Reds)
        
        plt.figure('XGBoost Classfier')
        plt.title('XGBoost Classfier')
        plt.xlabel('Classes')
        plt.ylabel('Classes')
        self.cm4=confusion_matrix(self.y_test,self.y_xgbc)
        sns.heatmap(self.cm4,annot=True,cmap=plt.cm.Reds)
        
        
    #%% Running Main Program
    def run(self):
        self.open_db()
        self.clean_df()
        self.handle_nan()
        self.seperate()
        self.reduce_dim()
        self.cluster()
        self.cv()
        self.batch()
        
        return self.dataframe,self.encoded_df,self.stats.T,\
               self.stats_2.T,self.x,self.y,self.filtered_df,\
               self.new_x_pca,self.pre_results

if __name__=="__main__":
    file="swiss_ha.sql"
    data=Data(file)
    df,new_df,stats,stats_2,x,y,filt,comp,pre_res=data.run()