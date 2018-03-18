import pandas as pd
import numpy as np

# for column
#df['column'] = df['column'].replace(np.nan, 0)

# for whole dataframe
yelp = pd.read_csv('women.csv')
#print(yelp.shape)
X = yelp['Review_Text']
y = yelp['Recommended']
print(y[9])
print(np.shape(X))
print(np.shape(y))

print(type(y))
#df = pd.read_csv('.csv')
y = y.replace(np.nan, 0)

# inplace
#df.replace(np.nan, 0, inplace=True)
df3 = pd.DataFrame({"Recommended" :y})  #write the cleaned training data to  new file for loading 
df3.to_csv("nancleared123.csv", index=False)
