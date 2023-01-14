#import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
#from sklearn import train_test_split 
#from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import RandomizedSearchCV

#@st.cache
#def loadData():
    flights = pd.read_csv('/home/ubuntu/flight_delaay/vs_flight/flights.csv')
    airport = pd.read_csv('/home/ubuntu/flight_delaay/vs_flight/airports.csv')

    variables_to_remove=["YEAR","FLIGHT_NUMBER","TAIL_NUMBER","DEPARTURE_TIME","TAXI_OUT","WHEELS_OFF","ELAPSED_TIME","AIR_TIME","WHEELS_ON","TAXI_IN","ARRIVAL_TIME","DIVERTED","CANCELLED","CANCELLATION_REASON","AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","RESULT","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]
    flights.drop(variables_to_remove,axis=1,inplace= True)

    flights.loc[~flights.ORIGIN_AIRPORT.isin(airport.IATA_CODE.values),'ORIGIN_AIRPORT']='OTHER'
    flights.loc[~flights.DESTINATION_AIRPORT.isin(airport.IATA_CODE.values),'DESTINATION_AIRPORT']='OTHER'

    flights=flights.dropna()

    df=pd.DataFrame(flights)
    df['DAY_OF_WEEK']= df['DAY_OF_WEEK'].apply(str)
    df["DAY_OF_WEEK"].replace({"1":"SUNDAY", "2": "MONDAY", "3": "TUESDAY", "4":"WEDNESDAY", "5":"THURSDAY", "6":"FRIDAY", "7":"SATURDAY"},inplace=True)

    dums = ['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DAY_OF_WEEK']
    df_cat=pd.get_dummies(df[dums],drop_first=True)

    var_to_remove=["DAY_OF_WEEK","AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]
    df.drop(var_to_remove,axis=1,inplace=True)

    data=pd.concat([df,df_cat],axis=1)
    final_data = data.sample(n=100000)
    #return final_data

#def preprocessing(final_data):
    X=final_data.drop("DEPARTURE_DELAY",axis=1)
    Y=final_data.DEPARTURE_DELAY
   # le=LabelEncoder()
 #   y= le.fit_transform(y.flatten())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
 #   return X_train,X_test,y_train,y_test,le

#def rfg():
    reg_rf = RandomForestRegressor()
    reg_rf=reg_rf.fit(X_train,y_train)
    y_pred = reg_rf.predict(X_test)
    reg_rf.score(X_train, y_train)
    reg_rf.score(y_test, y_pred)


def prediction(MONTH, DAY,SCHEDULED_DEPARTURE,
       DISTANCE, ARRIVAL_DELAY,AIRLINE,ORIGIN_AIRPORT,DESTINATION_AIRPORT,DAY_OF_WEEK):
    AIRLINE_index = np.where(X.columns==AIRLINE)[0][0]
    ORIGIN_index = np.where(X.columns==ORIGIN_AIRPORT)[0][0]
    DESTINATION_index = np.where(X.columns==DESTINATION_AIRPORT)[0][0]
    DAY_OF_WEEK_index = np.where(X.columns==DAY_OF_WEEK)[0][0]
    x= np.zeros(len(X.columns))
    x[0] = MONTH
    x[1] = DAY
    x[2] = SCHEDULED_DEPARTURE
    x[3] = DISTANCE
    x[4] = ARRIVAL_DELAY
    if AIRLINE_index >=0:
        x[AIRLINE_index] = 1
    if ORIGIN_index >=0:
        x[ORIGIN_index] = 1
    if DESTINATION_index >=0:
        x[DESTINATION_index] = 1
    if  DAY_OF_WEEK_index >= 0:
        x[ DAY_OF_WEEK_index] = 1    

    return reg_rf.predict([x])[0]

import streamlit as st
def main():
    st.title("Flight Delay Prediction")

    #final_data= loadData()

    #X_train,X_test,y_train,y_test,le= preprocessing(final_data)
    ##y_pred = reg_rf.predict(X_test)

    st.subheader("Prediction using Random Forest Regressor")
    text1= st.number_input("Enter month in digit",min_value=1,max_value=12)
    #text1= int(text1, base= 16)

    choice= st.selectbox("Prediction method",["Random Forest Regressor","GBR"])
    if st.button("Predict"):
        if choice=="Random Forest Regressor":
            res= prediction(text1,6,1515,328,-8.0,'AIRLINE_OO','ORIGIN_AIRPORT_PHX','DESTINATION_AIRPORT_ABQ','DAY_OF_WEEK_TUESDAY')
        st.write(res)
if __name__=='__main__':
    main()
    #print(res)

