import pandas as pd 
data = pd.read_csv('/home/kaustubh/Desktop/nlg2/nlg2/rubics_deploy/stockdata.csv')
data = data.iloc[:,-1:]
data.to_csv('labels.csv',index=False)