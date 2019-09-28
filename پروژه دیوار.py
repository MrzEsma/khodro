print('بسم الله الرحمن الرحیم')

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics.regression import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def cal_divar(year,milage,car,city):
    
    df=pd.read_csv('F:\data mining\dataset\divar_posts_dataset\divar_posts_dataset.csv')
    df_car=df.copy()[df.cat1=='vehicles']
    df_car=df_car.copy()[df_car.cat2=='cars']
    df_car=df_car.copy()[df_car.cat3=='light']
    df_car=df_car[['brand','city','year','mileage','price']]
    df_car=df_car.replace({'پژو ۲۰۶‍::Peugeot 206':'p206',
                                                                                   'پراید صندوق‌دار::Pride':'pride',
                                                                                   'پژو ۴۰۵::Peugeot 405':'p405',
                                                                                   'سایر':'other','سمند::Samand':'samand',
                                                                                   'پژو پارس::Peugeot Pars':'pars',
                                                                                   'پراید هاچ‌بک::Pride':'pride_hback',
                                                                                   'وانت':'Pickup','تیبا::Tiba':'tiba',
                                                                                   'ام‌وی‌ام::MVM':'mvm',
                                                                                   'پژو ۲۰۶‍ صندوق‌دار::Peugeot 206':'p206sd',
                                                                                   'نیسان::Nissan':'nissan',
                                                                                   'هیوندای (غیره)::Hyundai':'hyundai',
                                                                                   'پیکان::Peykan':'peykan',
                                                                                   'تندر ۹۰::Tondar 90':'L90','کیا::Kia':'kia',
                                                                                   'پژو روآ / آر‌دی::RD/ROA':'rd_roa',
                                                                                   'زانتیا::Citroen Xantia':'xantia',
                                                                                   'تویوتا::Toyota':'toyota','رنو::Renault':'reno',
                                                                                   'لیفان::Lifan':'lifan','دوو::Daewoo':'deawoo',
                                                                                   'بنز::Mercedes-Benz':'benz',
                                                                                   'رانا::Runna':'ranna','بی‌ام‌و::BMW':'bmw',
                                                                                   'هیوندای سوناتا::Hyundai Sonata':'sonata'})
    df_car.reset_index(inplace=True)
    del df_car['index']
    df_car=df_car[df_car['brand'].isin(['pride','pride_hback','p206','p405','pars','peykan','samand','Pickup'])]
    df_car=df_car.drop(df_car[df_car.price<2500000].index)
    df_car.replace({'<1366':'1350'},inplace=True)
    df_car['year']= df_car['year'].astype(str).astype(int)
    df_car=pd.get_dummies(df_car)

    x=df_car[['year', 'mileage', 'brand_Pickup', 'brand_p206', 'brand_p405',
           'brand_pars', 'brand_peykan', 'brand_pride', 'brand_pride_hback',
           'brand_samand', 'city_Ahvaz', 'city_Isfahan', 'city_Karaj',
           'city_Kermanshah', 'city_Mashhad', 'city_Qom', 'city_Shiraz',
           'city_Tabriz', 'city_Tehran']]
    y=df_car[['price']]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)

    reg=GradientBoostingRegressor(n_estimators=212,max_depth=5,warm_start=True)
    reg.fit(x_train,y_train.values.ravel())
    y_pred=reg.predict(x_test)
    print('r2 score: ',r2_score(y_test,y_pred),'\n')

    a=[]
    for i in range(19):
        a.append(0)

    a[0]=year
    a[1]=milage
    a[car+1]=1
    a[city+9]=1
    
    a=np.array(a)
    a=a.reshape(1,-1)
    c=2.3
    print('price is : ',int(reg.predict(a)*c))

#===========================================

y=input('year?')
m=input('milage?')
model=input('car model?')
c=input('your city?')

cal_divar(int(y),int(m),int(model),int(c))
