#!/usr/bin/env python
# coding: utf-8

# In[744]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
print(ipl_data.head())


# In[850]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
ba_before = ipl_data['Player batting average for previous season']
auction_year = ipl_data['Year']
player = ipl_data['Player Name']
auction_price = ipl_data['Auction price (US$)']
sr_before = ipl_data['Player Strike Rate on previous season']
fours_before = ipl_data['Player 4s on previous season']
sixes_before = ipl_data['Player 6s on previous season']
ba_after = ipl_data['Player batting average following auction']
sr_after = ipl_data['Player strike rate following auction']
fours_after = ipl_data['Player 4s following auction']
sixes_after = ipl_data['Player 6s following auction']
        
    
features = ipl_data[['Player 6s on previous season', 'Player batting average for previous season', 'Player Strike Rate on previous season', 'Player 4s on previous season']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)
print(model.score(features_test, outcomes_test))


# In[877]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')


ipl_data['impact_metric'] = ((ipl_data['Player batting average following auction'] + ipl_data['Player strike rate following auction']) / 100)+(ipl_data['Player 4s following auction'] + ipl_data['Player 6s following auction'] * 6)

auction_price = ipl_data['Auction price (US$)']
impact = ipl_data['impact_metric']


features = ipl_data[['impact_metric']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)
print(model.score(features_test, outcomes_test))


# In[835]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
ba_before = ipl_data['Player batting average for previous season']
auction_year = ipl_data['Year']
player = ipl_data['Player Name']
auction_price = ipl_data['Auction price (US$)']
sr_before = ipl_data['Player Strike Rate on previous season']
fours_before = ipl_data['Player 4s on previous season']
sixes_before = ipl_data['Player 6s on previous season']
ba_after = ipl_data['Player batting average following auction']
sr_after = ipl_data['Player strike rate following auction']
fours_after = ipl_data['Player 4s following auction']
sixes_after = ipl_data['Player 6s following auction']


plt.scatter(ba_before, auction_price, alpha=0.4, color='blue')
plt.scatter(sr_before, auction_price, alpha=0.4, color='red')
plt.title("Batting Average(blue) and Strike Rate(red) vs Auction Price")
plt.show()
plt.scatter(ba_after, auction_price, alpha=0.4, color='blue')
plt.scatter(sr_after, auction_price, alpha=0.4, color='red')
plt.show()
plt.scatter(sixes_before, auction_price, alpha=0.4)
plt.show()
plt.scatter(fours_before, auction_price, alpha=0.4)
plt.show()


# In[743]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
ba_before = ipl_data['Player batting average for previous season']
auction_year = ipl_data['Year']
player = ipl_data['Player Name']
auction_price = ipl_data['Auction price (US$)']
sr_before = ipl_data['Player Strike Rate on previous season']
fours_before = ipl_data['Player 4s on previous season']
sixes_before = ipl_data['Player 6s on previous season']
ba_after = ipl_data['Player batting average following auction']
sr_after = ipl_data['Player strike rate following auction']
fours_after = ipl_data['Player 4s following auction']
sixes_after = ipl_data['Player 6s following auction']
        
    
features = ipl_data[['Player 6s following auction', 'Player batting average following auction', 'Player strike rate following auction', 'Player 4s following auction']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)
print(model.score(features_test, outcomes_test))


# In[920]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
ba_before = ipl_data['Player batting average for previous season']
auction_year = ipl_data['Year']
player = ipl_data['Player Name']
auction_price = ipl_data['Auction price (US$)']
sr_before = ipl_data['Player Strike Rate on previous season']
fours_before = ipl_data['Player 4s on previous season']
sixes_before = ipl_data['Player 6s on previous season']
ba_after = ipl_data['Player batting average following auction']
sr_after = ipl_data['Player strike rate following auction']
fours_after = ipl_data['Player 4s following auction']
sixes_after = ipl_data['Player 6s following auction']
        
    
features = ipl_data[['Player batting average for previous season']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)
print(model.score(features_test, outcomes_test))


# In[1057]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')


ipl_data['impact_metric'] = (ipl_data['Player batting average following auction'] / 100)+(ipl_data['Player 4s following auction'] + ipl_data['Player 6s following auction'] * 1.5)

auction_price = ipl_data[['Auction price (US$)']]
impact = ipl_data[['impact_metric']]
model2 = LinearRegression()
model2.fit(impact, auction_price)

price_predict = model.predict(impact)
        
plt.scatter(impact, auction_price, alpha=0.4)
plt.scatter(0, 1750000, color='red')
plt.scatter(100, 0, color='green')
plt.plot(impact, price_predict)
plt.title('Impact Metric v. Player Auction Price')
plt.xlabel('Player Impact Metric Score')
plt.ylabel('Player Auction Price (US$)')
plt.show()


# In[1023]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')
ba_before = ipl_data['Player batting average for previous season']
auction_year = ipl_data['Year']
player = ipl_data['Player Name']
auction_price = ipl_data['Auction price (US$)']
sr_before = ipl_data['Player Strike Rate on previous season']
fours_before = ipl_data['Player 4s on previous season']
sixes_before = ipl_data['Player 6s on previous season']
ba_after = ipl_data['Player batting average following auction']
sr_after = ipl_data['Player strike rate following auction']
fours_after = ipl_data['Player 4s following auction']
sixes_after = ipl_data['Player 6s following auction']
  
    
features = ipl_data[['Player Strike Rate on previous season']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)
print(model.score(features_test, outcomes_test))


# In[1050]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

ipl_data = pd.read_csv(r'C:\Users\monik\Documents\CricketProject.csv')


ipl_data['impact_metric'] = (ipl_data['Player batting average following auction'] / 100)+(ipl_data['Player 4s following auction'] + ipl_data['Player 6s following auction'] * 1.5)
ipl_data['impact/price metric'] = (ipl_data['impact_metric'] * 100) / ipl_data['Auction price (US$)']
player = ipl_data['Player Name']

auction_price = ipl_data['Auction price (US$)']
impact = ipl_data['impact_metric']
impact_over_price_metric = ipl_data['impact/price metric']
features = ipl_data[['impact_metric']]
outcomes = ipl_data[['Auction price (US$)']]
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcomes_train)

y_pos = np.arange(len(player))

    
plt.figure(figsize=(10, 15))             
plt.barh(y_pos, impact_over_price_metric, alpha=0.4)

plt.title('Price-Impact Comparison Metric')
plt.xlabel('Player Impact Metric Score')
plt.ylabel('Player')
plt.yticks(y_pos, player)
plt.savefig('project_graph2.png')


# In[ ]:




