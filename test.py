import pandas as pd 
a = pd.read_csv('songdata.csv')
b = pd.read_csv('women.csv')
frames = [a,b]
c = pd.concat(frames)
#print(c.head())
#df3 = c
#df3 = pd.DataFrame({"free_fund" : df2.Observation, "Energy" :c})
c.to_csv("lala.csv", index=False)