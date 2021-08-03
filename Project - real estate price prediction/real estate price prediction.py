###########################################################

# REAL ESTATE PRICE PREDICTION 
# CSV USED: Bengaluru_House_data.csv from kaggle.com

###########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)



df1 = pd.read_csv("Bengaluru_House_Data.csv")

#dependent variable is Price (what I want to predict). Price is in indian rupees
#the others are the independent variables


df1.head()
df1.shape #to see the number of rows and columns (8 features and 1 target variable)

df1.groupby("area_type")["area_type"].count() #or agg.("count")

#now I can drop the columns that are not really important
#in deciding the final price
#I'm assuming that these columns are not useful in predicting the price

df2 = df1.drop(["area_type","society","balcony","availability"],axis=1)
df2.head()

#data cleaning process.. how do we handle the nan values?

df2.isnull().sum() #bath and size are particularly affected by nan values

#I can fill the nan values with the median, but since the dataset is pretty huge
#i can drop the rows where we have nan values

df3 = df2.dropna()

df3.isnull().sum() #we don't have nan values anymore

#let's analyze the size features,( BHK==bedroom)
df3["size"].unique()

#create a new column

df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))

df3["bhk"].unique() #now I only have numbers, not text anymore

df3[df3["bhk"] > 20] #I find 2 homes with this feature,
#but I can already tell that there is an error here:
# the total sqft in one home is 2400 and there are 43 bedrooms which is impossible


#let's then check the total_sqft feature

df3.total_sqft.unique() #I notice that there are numbers but there are also ranges
# we want to convert the range to a number 

def is_float(x):
    try:
        float(x) #try to convert into a float number 
    except:
        return False
    return True

df3[df3["total_sqft"].apply(is_float)]

df3["total_sqft"].dtypes

def convert_sqft_to_num(x):
    tokens= x.split("-")
    if len(tokens) == 2:
       return (float(tokens[0])+ float(tokens[1]))/2
    try:
       return float(x)
    except:
       return None        


#let's test this function

convert_sqft_to_num("1200") #return the float number

convert_sqft_to_num("1200-1201") #return the mean

convert_sqft_to_num("1200m") #return None

df4 = df3.copy()

df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
#I can check some values

df4.iloc[30] 
df4.iloc[0] #first row



df4.isnull().sum() #we have still 46 nan values that I need to get rid off



df5 = df4.copy()
#we create a new feature, which will be very useful for outlier detection 

df5["price_per_sqft"] = df5["price"]*100000/df5["total_sqft"]


#explore the location column


len(df5["location"].unique()) #we have 1304 locations which is a lot
#it's hard to get the dummy variables since there are too many locations
#too many features
#DIMENSIONALITY CURSE
#there are techniques to reduce the dimension

#how many datapoints for locations?

df5.location = df5.location.apply(lambda x: x.strip()) #remove any space

location_stats= df5.groupby("location")["location"].count().sort_values(ascending=False)
#there are many locations with only one data point

#so any location that has, for example,less than 10 data points is called "other location


len(location_stats[location_stats <=10]) #1052 points

location_stats_less_than_10 = location_stats[location_stats <=10] 
#put them in a general category


len(df5.location.unique())

df5.location = df5.location.apply(lambda x:  "other" if x in location_stats_less_than_10 else x )
len(df5.location.unique())
#this is pretty good
#so for onw hot encoding I will only have 242 columns as dummy variables

#next step: OUTLIER DETECTION: they represent the extreme variation in a dataset
#although they are valid, it makes sense to remove them, 'cause outliers can create issues later on

#we can apply different techniques like simple domain knowledge

#first look the any data rows where the square foot per bedrooms is less than some treshold

#usually it's 300 square feets for bedrooms

#let's check where this treshold is not met

len(df5[df5.total_sqft/df5.bhk < 300]) #I only have 5 rows with this characteristic
len(df5[df5.total_sqft/df5.bhk >= 300])
#these data points are really unsual, these are data errors

df6 = df5[-(df5.total_sqft/df5.bhk < 300)] #use the negate symbol: -

df6.shape

df6.isnull().sum()

#now let's check price per square feet:  to detect outliers

df6.price_per_sqft.describe()
df6.groupby("location").count()
#remove the outliers: per location I want to find the mean and standard deviation
#and then filter the datapoint that are beyond 1 standard deviation

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m= np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))] #I will keep this data points
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)

df7.shape #12502-10241 = 2216 datapoints removed

def plot_scatter_chart(df,location):
    bhk2= df[(df.location==location) & (df.bhk==2)]
    bhk3= df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams["figure.figsize"] == (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color="blue",label="2 BHK", s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color="green",marker="+",label="3 BHK",s=50)
    plt.xlabel=("total square feet area")
    plt.ylabel("price")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df7,"Rajaji Nagar")

#in the center of the graph there are outliers! we see that in the same location, some apartment
#with 3 bedrooms are cheaper than apartment with 2 bedrooms 

plot_scatter_chart(df7,"Hebbal")

#also in this location I find outliers

#I want to do some data cleaning

def remove_bhk_outliers(df): 
     exclude_indices = np.array([])
     for location, location_df in df.groupby("location"):
         bhk_stats= {}
         for bhk,bhk_df in location_df.groupby("bhk"):
             bhk_stats[bhk] = {
                 "mean": np.mean(bhk_df.price_per_sqft),
                 "std": np.std(bhk_df.price_per_sqft),
                 "count" : bhk_df.shape[0]
                 }
         for bhk, bhk_df in location_df.groupby("bhk"):
             stats = bhk_stats.get(bhk-1)
             if stats and stats["count"] >5:
                 exclude_indices =np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values)
                 
     return df.drop(exclude_indices,axis="index")          

df8 = remove_bhk_outliers(df7)


df8.shape #(7329,7)

#now I can again plot the same scatterplot to see what kind of improvement it has done
plot_scatter_chart(df8,"Hebbal")
#it worked, now all the outiers are removed

#how to plot the istogram:

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft, rwidth= 0.8)
plt.xlabel("price per square feet")
plt.ylabel("Count")

#majority of the data points are in the 50000 price per square feet
#kinda looks like a normal distribution

#let's explore the bathroom feature

df8.bath.unique()

df8[df8.bath>10] #place per we have more than 10 bathrooms

#usually if number of bathrooms > number of bedrooms + 2, we are going to remove those data points

plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("number of bathrooms")
plt.ylabel("count")

df8[df8.bath>df8.bhk+2] #I want to remove these outliers
df9 =df8[df8.bath< df8.bhk+2]
df9.shape

#now the dataset is neat and clean, I can prepare for machine learning training
#let's remove the not useful features

#price per square foot can be removed, size can be removed 

df10 = df9.drop(["size","price_per_sqft"],axis=1)

df10.shape
df10.head(3)

#MODEL BUILDING: we will find the best algorithm and the best parameters
#the dataframe we are using is the df10.

#price is the target variable
#we need to convert the location into a numeric column
#ONE HOT ENCODING


pd.get_dummies(df10.location)

dummies= pd.get_dummies(df10.location)


df11 = pd.concat([df10,dummies],axis=1)

#to avoid dummy variaible trap let's remove one dummy column

df11 = df11.drop("other",axis=1)

df12 = df11.drop("location",axis=1)

df12.head(3)
df12.shape
X = df12.drop("price",axis=1)
#let's start building our model

y= df12.price

y.head()
X.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

len(X_test)
len(X_train)

#it's a regression problem rather than a classification problem, we are predicting the price

from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

#now we are using the K-fold cross validation to see how much we can increase the score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit ( n_splits=5,test_size=0.2,random_state=42) #will randomize each sample


scores = cross_val_score(LinearRegression(),X,y,cv=cv)

np.mean(scores)


#let's try other regression techniques, to try figure out which one give me the best scores

from sklearn.model_selection import GridSearchCV

#let's try lasso and decision tree regression

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

#hyper parameters tuning
def find_best_model_using_gridSearchcv(X,y):
      algos= {
     
            "linear_regression": {
            "model": LinearRegression(),
            "params": {
                "normalize": [True,False]
                }
            },
            "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [1,2],
                "selection": ["random","cyclic"]
                }
            },
            "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["mse","fredman_mse"],
                "splitter": ["best","random"]
                }
             }
       }
      scores= []
      cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=42)
      for algo_name,config in algos.items():
            gs = GridSearchCV(config["model"],config["params"],cv=cv,return_train_score=False)
            gs.fit(X,y)
            scores.append({
                "model": algo_name,
                "best_score": gs.best_score_,
                "best_params": gs.best_params_
                })
      return pd.DataFrame(scores,columns=["model","best_score","best_params"])

find_best_model_using_gridSearchcv(X,y)

#the best is still linear regression, with normalize:True

#so I will use linear regression to predict price,since it is still the best model 
np.where(X.columns=="2nd Phase Judicial Layout")[0][0]



def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0] #gives the column index
    
    x = np.zeros(len(X.columns))
    x[0] =sqft
    x[1] =bath
    x[2] =bhk
    if loc_index >=0:
        x[loc_index] =1
    return lr_clf.predict([x])[0]  

predict_price("1st Phase JP Nagar",1000,2,2) #it will predict the price 

predict_price("1st Phase JP Nagar",1000,3,3) 


predict_price("1st Phase JP Nagar",1000,3,3)

predict_price("Indira Nagar",1000,2,2) #most expensive

predict_price("Indira Nagar",1000,3,3)
 
#our data is telling us that a 3 bedrooms apartment they cost less for 2 bedrooms apartment
#maybe more bedrooms mean they are smaller, so it has less value

#now export the model to a PICKLE file, so other can use it
#simple code:
import pickle
with open("banglore_home_prices_model.pickle","wb") as f:
    pickle.dump(lr_clf,f)


import json
columns= {
       "data_columns": [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
   f.write(json.dump(columns))

#now we will build python flask server

#END














