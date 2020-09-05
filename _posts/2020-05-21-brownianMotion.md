---
title: "Simulating Brownian Motion"
date: 2020-05-21
tags: [statistics, modelling]
excerpt: "Simulating Brownian Motion and using it to model stock prices"
mathjax: "true"
---
### Simulating Brownian Motion

Brownian Motion is the seemingly random movement of particles suspended in a fluid (i.e. dust motes in water or air) and by extension the mathematical model used to describe this movement. It is one of the simplest and most fundamental continuous-time
stochastic processes, finding applications in numerous situations.
A stochastic process <img src="https://latex.codecogs.com/gif.latex?X(t),&space;t&space;\geq&space;0" title="X(t), t \geq 0" /> is Brownian if: <br/>
<img src="https://latex.codecogs.com/gif.latex?X(0)&space;=&space;0" title="X(0) = 0" /> (arbitrary choice)<br/>
<img src="https://latex.codecogs.com/gif.latex?X(t),&space;t&space;\geq&space;0" title="X(t), t \geq 0" /> has stationary and independant increments<br/>
and for every <img src="https://latex.codecogs.com/gif.latex?t&space;>&space;0,&space;X(t)&space;\sim&space;\mathcal{N}(0,&space;\sigma^2t)" title="t > 0, X(t) \sim \mathcal{N}(0, \sigma^2t)" />

Thus it follows, for any <img src="https://latex.codecogs.com/gif.latex?t,&space;s&space;>&space;0" title="t, s > 0" />:

$$X(s+t)|X(s)=x) \sim \mathcal{N}(x,\,\sigma^{2}t)$$

When <img src="https://latex.codecogs.com/gif.latex?\sigma&space;=&space;1" title="\sigma = 1" /> the process is known as Standard Brownian Motion 


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style

def Brownian(u, sigma, dt, moves):
    
    #dt: time increment
    #moves: number of increments, dt*moves = total time of process
    
    random_movements = np.random.normal(u, dt*sigma**2, moves)
    B_t = np.cumsum(random_movements)
    
    #Plot
    t = np.linspace(0, dt*moves, moves)
    
    return B_t, t
```


```python
for i in range(0, 5):
    B_t, t = Brownian(0, 1, 1, 100)
    plt.step(t, B_t)
    
style.use('seaborn-dark-palette')
plt.title("Standard Brownian Motion")
plt.xlabel("t", fontsize=16)  
plt.ylabel("B(t)", fontsize=16)  
plt.show()
```


![](/images/BM_outputs/output_3_0.png)


Brownian Motion with drift coefficient <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /> and variance parameter <img src="https://latex.codecogs.com/gif.latex?\sigma^2" title="\sigma^2" />:

$$X(t) \sim \mathcal{N}(\mu t, \sigma^2t)$$


```python
for i in range(0, 5):
    B_t, t = Brownian(-1.5, 2, 1, 100)
    plt.step(t, B_t)
    
plt.title("Brownian Motion with Drift")
plt.xlabel("t", fontsize=16)  
plt.ylabel("B(t)", fontsize=16)  
plt.show()
```


![](/images/BM_outputs/output_6_0.png)


### Simulating Stock Prices

Google stock price data


```python
import pandas as pd

#Google Stock price from 2013-02-08 - 2018-02-07
df = pd.read_csv("GOOGL_data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-02-08</td>
      <td>390.4551</td>
      <td>393.7283</td>
      <td>390.1698</td>
      <td>393.0777</td>
      <td>6031199</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-02-11</td>
      <td>389.5892</td>
      <td>391.8915</td>
      <td>387.2619</td>
      <td>391.6012</td>
      <td>4330781</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-02-12</td>
      <td>391.2659</td>
      <td>394.3440</td>
      <td>390.0747</td>
      <td>390.7403</td>
      <td>3714176</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-02-13</td>
      <td>390.4551</td>
      <td>393.0677</td>
      <td>390.3750</td>
      <td>391.8214</td>
      <td>2393946</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-02-14</td>
      <td>390.2549</td>
      <td>394.7644</td>
      <td>389.2739</td>
      <td>394.3039</td>
      <td>3466971</td>
      <td>GOOGL</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.step(df["date"], df["close"])
plt.xticks(range(0, len(df), int(len(df)/5)), ["2013", "2014", "2015", "2016", "2017", "2018"])
plt.title("Google Closing Price")
plt.xlabel("Day", fontsize=16)  
plt.ylabel("Share Price", fontsize=16) 
plt.show()
```


![](/images/BM_outputs/output_10_0.png)


We can see below that the daily returns are approximately normally distributed,


```python
#calculate daily returns
daily_returns = []
net_daily_returns = []
for index in range(0, len(df) - 1):
    daily_returns.append((df.loc[index + 1]["close"] - df.loc[index]["close"]) / df.loc[index]["close"])
    net_daily_returns.append((df.loc[index + 1]["close"] - df.loc[index]["close"]))

plt.hist(daily_returns, bins = 100, density = True)
plt.title("Historgram of Percentage Returns")
plt.show() 
print("Mean: ", np.mean(daily_returns))
print("Std Deviation: ", np.std(daily_returns))
```


![](/images/BM_outputs/output_12_0.png)


    Mean:  0.0008800658343619005
    Std Deviation:  0.013874115811593625
    

We can try to model using the mean and variance of the daily_returns, however this is not very accurate.


```python
for i in range(0, 5):
    B_t, t = Brownian(np.mean(net_daily_returns), np.std(net_daily_returns), 1, len(net_daily_returns) + 1)
    plt.step(t, B_t)

plt.xlabel("t", fontsize=16)  
plt.ylabel("S(t)", fontsize=16)     
plt.show()
```


![](/images/BM_outputs/output_14_0.png)


Random fluctuations in stock price over a shprt period of time can be modelled using [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion). A stochastic process <img src="https://latex.codecogs.com/gif.latex?S(t)" title="S(t)" /> is said to follow a GBM if it satisfies the following stochastic differential equation:

$$dS(t) = \mu S(t)dt + \sigma S(t)dB(t)$$

Where <img src="https://latex.codecogs.com/gif.latex?\mu&space;S(t)dt" title="\mu S(t)dt" /> is a predictable / anticipated component of the expected return and <img src="https://latex.codecogs.com/gif.latex?\sigma&space;S(t)dB(t)" title="\sigma S(t)dB(t)" /> which is the stochastic / random component.
The solution to this equation <img src="https://latex.codecogs.com/gif.latex?S(t)" title="S(t)" /> follows the log-normal distribution (intuitive, stock prices cannot be negative and are normally distributed):

$$S(t) = S(0)exp[(\mu - \frac{\sigma^2}{2})t+\sigma B(t)]$$


The components of the exponent can be interpreted as drift and volatility/diffusion respectively.<br/>
Thus we can attempt to simulate Googles stock price:


```python
def GeoBrownian(S_0, u, sigma, dt, moves):

    t = np.linspace(0, dt*moves, moves)
    
    #Calculate Drift
    drift = (u - (sigma**2) / 2) * t

    #Calculate Volatility, each day has an element of randomness following Brownian Motion
    B_t = []
    for time in t:
        B_t.append(np.random.normal(0, dt)) 
    volatility = sigma * np.asarray(B_t)

    S_t = S_0 * np.exp(drift + volatility)
    return S_t, t
```


```python
from sklearn.metrics import r2_score

S_0 = df["close"][0] #Starting Price
moves = len(df) #Num Days
dt = 1 #Period of day
u = np.mean(daily_returns)
sigma = np.std(daily_returns)

i = 4 #Num Simulations

for x in range(0, i):
    S_t, t = GeoBrownian(S_0, u, sigma, dt, moves)
    plt.step(t, S_t)
    print("R2 Score: ", r2_score(df["close"], S_t))

plt.xlabel("t", fontsize=16)  
plt.ylabel("S(t)", fontsize=16) 
plt.title("{} Simulated Share Prices".format(i), fontsize=16)
plt.show()
```

    R2 Score:  0.9296291894967975
    R2 Score:  0.9276290857401792
    R2 Score:  0.9284940002464735
    R2 Score:  0.9288627107515467
    


![](/images/BM_outputs/output_17_1.png)


Over a short period of time this method is far less accurate


```python
i = 4 #Num Simulations
days = 100 

for x in range(0, i):
    S_t, t = GeoBrownian(S_0, u, sigma, dt, days)
    plt.step(t, S_t)
    print("R2 Score: ", r2_score(df["close"][:days], S_t))

plt.xlabel("t", fontsize=16)  
plt.ylabel("S(t)", fontsize=16)  
plt.title("{} Simulated Share Prices".format(i), fontsize=16)
plt.show()
```

    R2 Score:  0.2159292025995374
    R2 Score:  0.1782694591687074
    R2 Score:  0.20872809262978953
    R2 Score:  0.23681712790377907
    


![](/images/BM_outputs/output_19_1.png)


R2 against number of days modelled, compared with actual Google stock data.


```python
r2_scores = []
for day in range(2, len(df)):
    S_t, t = GeoBrownian(S_0, u, sigma, dt, day)
    r2_scores.append(r2_score(df["close"][:day], S_t))

plt.plot(range(102, len(df)), r2_scores[100:])
plt.xlabel("Day", fontsize=16)  
plt.ylabel("R2", fontsize=16)  
plt.show()

plt.step(df.index[100:], df["close"][100:])
plt.step(df.index[100:], S_t[99:], label = "Simulation")
plt.xlabel("Day", fontsize=16)  
plt.ylabel("Share Price", fontsize=16) 
plt.legend(loc='best')
plt.show()
```


![](/images/BM_outputs/output_21_0.png)



![](/images/BM_outputs/output_21_1.png)


We can run a training and test split on the data


```python
def testTrainGeoBrownian(train_length, daily_return, closing_price, num_simulations):
    
    x = int(len(daily_return)*0.8)
    
    x_train = daily_return[:x]

    x_test = closing_price[:x]
    y_test = closing_price[x:len(daily_return)] #dont use last value of closing price

    i = num_simulations #Num Simulations on training data to run
    train_R2s = []
    test_R2s = []

    #Train models, model will extrapolate out to total number of days
    for j in range(0, i):
        S_t, t = GeoBrownian(closing_price[0], np.mean(x_train), np.std(x_train), 1, len(daily_return))

        x_preds = S_t[:x]
        y_preds = S_t[x:]
        
        #Calculate training score for current simulation
        train_R2s.append(r2_score(x_test, x_preds))
        #Calculate test score for current simulation
        test_R2s.append(r2_score(y_test, y_preds))

    print("Training R2 Score over {} iterations: ".format(i), np.mean(train_R2s))
    print("Test R2 Score over {} iterations: ".format(i), np.mean(test_R2s))
```


```python
train_length = 0.8 #length of training data
daily_return = daily_returns #Array of daily returns
closing_price = df["close"] #Array of closing prices
num_simulations = 4

testTrainGeoBrownian(train_length, daily_return, closing_price, num_simulations)
```

    Training R2 Score over 4 iterations:  0.8222586900633191
    Test R2 Score over 4 iterations:  0.1929507465148583
    

Thus the model is poor at generalizing to unseen data. Low Bias, High Variance. 

S&P/ASX 200 Data


```python
axjo = pd.read_csv("^AXJO.csv")
axjo.dropna(inplace = True)
axjo = axjo.reset_index()
axjo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2012-01-03</td>
      <td>4069.899902</td>
      <td>4108.100098</td>
      <td>4069.100098</td>
      <td>4101.200195</td>
      <td>4101.200195</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2012-01-04</td>
      <td>4124.100098</td>
      <td>4202.399902</td>
      <td>4123.899902</td>
      <td>4187.799805</td>
      <td>4187.799805</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2012-01-05</td>
      <td>4179.399902</td>
      <td>4179.399902</td>
      <td>4131.700195</td>
      <td>4142.700195</td>
      <td>4142.700195</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2012-01-06</td>
      <td>4140.299805</td>
      <td>4144.100098</td>
      <td>4107.799805</td>
      <td>4108.500000</td>
      <td>4108.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2012-01-09</td>
      <td>4115.399902</td>
      <td>4129.000000</td>
      <td>4088.699951</td>
      <td>4105.399902</td>
      <td>4105.399902</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.step(axjo["Date"], axjo["Close"])
plt.xticks(range(0, len(axjo), int(len(axjo)/8)), ["2012","2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"])
plt.title("S&P/ASX 200 Closing Price")
plt.xlabel("Day", fontsize=16)  
plt.ylabel("Share Price", fontsize=16) 
plt.show()
```


![](/images/BM_outputs/output_28_0.png)



```python
axjo_daily_returns = []
for index in range(0, len(axjo) - 1):
    axjo_daily_returns.append((axjo.loc[index + 1]["Close"] - axjo.loc[index]["Close"]) / axjo.loc[index]["Close"])
    
plt.hist(axjo_daily_returns, bins = 100, density = True)
plt.title("S&P/ASX 200 Percentage Daily Returns")
plt.show() 
print("Mean: ", np.mean(axjo_daily_returns) * 100, "%")
print("Std Deviation: ", np.std(axjo_daily_returns) * 100, "%")
```


![](/images/BM_outputs/output_29_0.png)


    Mean:  0.027254460238958996 %
    Std Deviation:  0.7829545680256039 %
    


```python
S_t, t = GeoBrownian(axjo["Close"][0], np.mean(axjo_daily_returns), np.std(axjo_daily_returns), 1, len(axjo_daily_returns))
plt.step(t, S_t)
print("R2 Score: ", r2_score(axjo["Close"][:2020], S_t))
plt.step(t, axjo["Close"][:-1])
plt.xlabel("t", fontsize=16)  
plt.ylabel("S(t)", fontsize=16) 
plt.title("Share Price of ^AXJO".format(i), fontsize=16)
plt.show()
```

    R2 Score:  0.6029624981291966
    


![](/images/BM_outputs/output_30_1.png)


As can be seen above the model is much worse at simulating AXJO data when compared with GOOGL data. This is due to the std deviation (volatility) of each data set:


```python
print("GOOGL Sdt: ", np.std(df["close"]))
print("^AXJO Sdt: ", np.std(axjo["Close"]))
```

    GOOGL Sdt:  187.499383851121
    ^AXJO Sdt:  638.2199355359211
    


```python
testTrainGeoBrownian(0.75, axjo_daily_returns, axjo["Close"], 4)
```

    Training R2 Score over 4 iterations:  0.2935100400352214
    Test R2 Score over 4 iterations:  0.4443542099722414
    

As can be seen from the scores this simulation generalizes well but is overall very poor.

### Conclusion

The GBM shows less accuracy for shorter time periods, however is capable of simulating the behaviour of a stock price over time. Improvements to this model could include treating the drift and volatility over time as stochastic and adapting model to external contextual factors.


