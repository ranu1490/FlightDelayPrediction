import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
#from sklearn import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import RandomizedSearchCV

@st.cache
def loadData():
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
    return final_data

def preprocessing(final_data):
    X=final_data.drop("DEPARTURE_DELAY",axis=1)
    Y=final_data.DEPARTURE_DELAY
   # le=LabelEncoder()
   # y= le.fit_transform(y.flatten())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train,X_test,y_train,y_test

def rfg(X_train, X_test,y_train,y_test):
    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train,y_train)
    #y_pred = reg_rf.predict(X_test)
    #score= accuracy_score(y_test,y_pred)*100
   # report= classification_report(y_test, y_pred)
    return reg_rf

def accept_data():
    month = st.number_input("Enter month ",min_value=1,max_value=12)
    day = st.number_input("Enter day",min_value=1,max_value=7)
    sch_dept = st.number_input("Enter scheduled deptaure")
    distance = st.number_input("Enter distance")
    arrival_delay = st.number_input("Enter arrival delay")
    airline = st.text_area("Enter airline code","Enter here")
    airline = 'AIRLINE_'+ airline 
    origin = st.text_area("Enter origin airport code","Enter here")
    origin = 'ORIGIN_AIRPORT_'+ origin
    destination = st.text_area("Enter destination airport code","Enter here")
    destination = 'DESTINATION_AIRPORT_'+destination
    day_of_week = st.text_area("Enter day of week","Enter here")
    day_of_week = 'DAY_OF_WEEK_'+ day_of_week

    return month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week
    
#import streamlit as st
def main():
    st.title("Flight Delay Prediction")

    final_data= loadData()

    X_train,X_test,y_train,y_test= preprocessing(final_data)
    ##y_pred = reg_rf.predict(X_test)

    st.subheader("Prediction using Random Forest Regressor")
  #  month= st.number_input("Enter month in digit",min_value=1,max_value=12)
    #text1= int(text1, base= 16)

    choice= st.slidebar.selectbox("CHOOSE ML MODEL",["Random Forest Regressor","GBR"])
    #if st.button("Predict"):
    if(choice=="Random Forest Regressor"):
        reg_rf = rfg(X_train,X_test, y_train,y_test)
        #st.text("Accuracy of random forest regressor is: ")
        #st.write(score,"%")
        try:
            if(st.checkbox("Want to predict on our own input?")):
                    #month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week
                user_data= accept_data()
                #res= prediction(month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week)
                res = reg_rf.predict(user_data) 
                if(res>=0):
                    st.write("Flight is not delayed")
                elif(res>= -15):
                    text1= "Flight is only delayed by "+str(res)+". Delays upto 15 minutes are taken into consideration"
                    st.write(text1)
                else:
                    text2= "Flight is delayed by "+str(res)+". Delays by more than 15 minutes are considered to be actual DELAYS"
                    st.write(text2)
        except:
            pass
if __name__=='__main__':
    main()
    #print(res)

