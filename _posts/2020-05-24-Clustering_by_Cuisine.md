---
title: "Clustering by Cuisine"
date: 2020-05-24
tags: [data visualisation, machine learning, api]
header:
  image: "/images/kayla-koss-WSbd7BELXak-unsplash.jpg"
excerpt: "Clustering of Cities based on their range of Cuisines"
mathjax: "true"
---
### Introduction
A key aspect when deciding where to travel are the cuisines and cultures of the destination. The aim of this project is to quantify the range of cuisines available in the most popular cities around the world, thus allowing travelers to make an informed decision about their next destination. Using machine learning techniques cities will be clustered based on the similarity of their cuisines.

### Data Collection

Data for the location of each city centre was sourced using the geopy package. 

```python
import requests #
import pandas as pd 
import numpy as np 
from geopy.geocoders import Nominatim
```


```python
#list of cities to cluster
cities = ["Tokyo", "Delhi", "Cairo", "Mexico City", "New York", "Los Angeles", "London", "Berlin",
         "Sydney", "Melbourne", "Paris", "Rome", "Seoul", "Lisbon", "Barcelona", "Moscow"]
lat = []
long = []

for city in cities:
    
    geolocator = Nominatim(user_agent="foursquare_agent")
    location = geolocator.geocode(city)
    lat.append(location.latitude)
    long.append(location.longitude)

data = pd.DataFrame({"City":cities, "Lat": lat, "Long": long}) 
data.sort_values("City", ascending = True, inplace = True)
data.head()
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
      <th>City</th>
      <th>Lat</th>
      <th>Long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Berlin</td>
      <td>52.517037</td>
      <td>13.388860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cairo</td>
      <td>30.048819</td>
      <td>31.243666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Delhi</td>
      <td>28.651718</td>
      <td>77.221939</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Lisbon</td>
      <td>38.707751</td>
      <td>-9.136592</td>
    </tr>
  </tbody>
</table>
</div>

Data for each city's most popular restaurant venue and range of cuisines will be sourced using the Foursquare API within python. 16 of the most popular travelling destinations were chosen for the cities. For each city the top 50 restaurant venues were located within 10km of the city centre. This data will then be used to group cities into clusters based on the similarity of their cuisines.

```python
#Foursquare Credentials
CLIENT_ID = #hidden
CLIENT_SECRET = #hidden
VERSION = '20180604'
LIMIT = 100
RADIUS = 10000
search_query = 'restaurant'

venues_list=[]
for name, lat, long in zip(data.City, data.Lat, data.Long):
    #Search for the top 50 resturants within 10km of the city centre
    url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(
        CLIENT_ID, 
        CLIENT_SECRET, 
        lat, 
        long, 
        VERSION, 
        search_query,
        radius, 
        LIMIT)
    
    #make request
    results = requests.get(url).json()["response"]["venues"]
    
    #get relevant data
    for venue in results:
        if (len(venue["categories"]) == 0):
            cuisine = None;
        else:
            cuisine = venue["categories"][0]["shortName"]
        
        #Some restaurants have cuisine unnamed
        if (cuisine == "Restaurant"):
            cuisine = None;
        
        venues_list.append([(
            name, 
            lat, 
            long, 
            venue['name'], 
            cuisine)])

city_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
city_venues.columns = ['City', 
                  'City Latitude', 
                  'City Longitude', 
                  'Venue', 
                  'Cuisine']
    
```


```python
city_venues
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
      <th>City</th>
      <th>City Latitude</th>
      <th>City Longitude</th>
      <th>Venue</th>
      <th>Cuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
      <td>Habibi Restaurant</td>
      <td>Halal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
      <td>Bar Restaurant Cervantes</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
      <td>Carballeira Restaurant</td>
      <td>Seafood</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
      <td>Restaurant CentOnze</td>
      <td>Spanish</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barcelona</td>
      <td>41.382894</td>
      <td>2.177432</td>
      <td>Arabia Bar &amp; Restaurant</td>
      <td>Moroccan</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Tokyo</td>
      <td>35.682839</td>
      <td>139.759455</td>
      <td>restaurant bar JAM</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>796</th>
      <td>Tokyo</td>
      <td>35.682839</td>
      <td>139.759455</td>
      <td>RESTAURANT OSURI</td>
      <td>Korean</td>
    </tr>
    <tr>
      <th>797</th>
      <td>Tokyo</td>
      <td>35.682839</td>
      <td>139.759455</td>
      <td>Restaurant Ito (レストラン・イト)</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>798</th>
      <td>Tokyo</td>
      <td>35.682839</td>
      <td>139.759455</td>
      <td>Chojyomen Sharks Fin Restaurant</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Tokyo</td>
      <td>35.682839</td>
      <td>139.759455</td>
      <td>Restaurant Verde (れすとらん べるで)</td>
      <td>Japanese Curry</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 5 columns</p>
</div>

Counting the number of restaurant venues obtained from each city. 

```python
city_venues.groupby("City").count()["Cuisine"]
```




    City
    Barcelona      36
    Berlin         34
    Cairo          43
    Delhi          43
    Lisbon         38
    London         35
    Los Angeles    47
    Melbourne      46
    Mexico City    33
    Moscow         32
    New York       48
    Paris          45
    Rome           43
    Seoul          49
    Sydney         48
    Tokyo          44
    Name: Cuisine, dtype: int64


Note that some restaurants did not have a cuisine associated with them, thus each city has less than 50 restaurant venues.<br.
<br>
Counting the number of unique cuisines obtained:
```python
print('{} unique cuisines.'.format(len(city_venues['Cuisine'].unique())))
```

    114 unique cuisines.


### Methodology

Firstly one-hot-encoding was performed on the city venue data and then grouped the data by city to obtain the mean of each cuisine within that city. 

```python
#One hot encoding
cities_onehot = pd.get_dummies(city_venues[['Cuisine']], prefix="", prefix_sep="")

#Add city column back
cities_onehot['City'] = city_venues['City'] 
fixed_columns = [cities_onehot.columns[-1]] + list(cities_onehot.columns[:-1])
cities_onehot = cities_onehot[fixed_columns]

#Group cities and take mean of frequency
city_cuisines = cities_onehot.groupby('City').mean().reset_index()
city_cuisines
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
      <th>City</th>
      <th>African</th>
      <th>American</th>
      <th>Arepas</th>
      <th>Asian</th>
      <th>Australian</th>
      <th>Austrian</th>
      <th>B &amp; B</th>
      <th>BBQ</th>
      <th>Bar</th>
      <th>...</th>
      <th>Tibetan</th>
      <th>Turkish</th>
      <th>Vegetarian / Vegan</th>
      <th>Vietnamese</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings</th>
      <th>Yemeni Restaurant</th>
      <th>Yoshoku</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barcelona</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berlin</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cairo</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Delhi</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lisbon</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>London</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Los Angeles</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Melbourne</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mexico City</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Moscow</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>New York</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Paris</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rome</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Seoul</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sydney</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tokyo</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 114 columns</p>
</div>

Now we can plot each cuisine with the normalized frequency it appears in all of the countries.

```python
import matplotlib.pyplot as plt

x = city_cuisines.sum()[1:]
y = city_cuisines.columns[1:]

#Normalize between 0 and 1
for i, val in enumerate(x):
    x[i] = (val - min(x)) / (max(x) - min(x))

plt.barh(y, x)
plt.rcParams["figure.figsize"]=20,20
plt.show()
```


![](/images/ClusteringCuisinesBarh.png)

At this stage k-means clustering can be employed in order to group the cities into 6 clusters based on the similarity of their cuisines. A ranking of the top 5 most popular cuisines for each city is also included.

```python
#Number of rankings
venues = 5

# create columns according to number of top venues
columns = ["City", "1", "2", "3", "4", "5"]

# create a new dataframe
top_cuisines = pd.DataFrame(columns = columns)
top_cuisines['City'] = city_cuisines['City']

for i in range(0, len(top_cuisines)):
    row = city_cuisines.iloc[i, :]
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    top_cuisines.iloc[i, 1:] = row_categories_sorted.index.values[0:venues]
```


```python
from sklearn.cluster import KMeans

#Get clusters
k = 6
kmeans = KMeans(n_clusters = k, random_state=0).fit(city_cuisines.drop("City" , 1))

#Insert Clusters into our ranked cuisine data
top_cuisines.insert(0, 'Cluster Labels', kmeans.labels_)
```


```python
top_cuisines
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
      <th>Cluster Labels</th>
      <th>City</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Barcelona</td>
      <td>Spanish</td>
      <td>Mediterranean</td>
      <td>Chinese</td>
      <td>Café</td>
      <td>Seafood</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Berlin</td>
      <td>German</td>
      <td>Breakfast</td>
      <td>French</td>
      <td>Hotel Bar</td>
      <td>Modern European</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Cairo</td>
      <td>Middle Eastern</td>
      <td>Falafel</td>
      <td>Yemeni Restaurant</td>
      <td>African</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Delhi</td>
      <td>Indian</td>
      <td>Chinese</td>
      <td>North Indian</td>
      <td>Diner</td>
      <td>Karaoke</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Lisbon</td>
      <td>Portuguese</td>
      <td>Indian</td>
      <td>Asian</td>
      <td>Himalayan</td>
      <td>Breakfast</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>London</td>
      <td>Chinese</td>
      <td>Indian</td>
      <td>English</td>
      <td>Italian</td>
      <td>Hotel Bar</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>Los Angeles</td>
      <td>Mexican</td>
      <td>Chinese</td>
      <td>Japanese</td>
      <td>Food</td>
      <td>American</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>Melbourne</td>
      <td>Chinese</td>
      <td>Korean</td>
      <td>Indian</td>
      <td>Asian</td>
      <td>Thai</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>Mexico City</td>
      <td>Mexican</td>
      <td>Bar</td>
      <td>Buffet</td>
      <td>Chinese</td>
      <td>American</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>Moscow</td>
      <td>Russian</td>
      <td>French</td>
      <td>Eastern European</td>
      <td>Modern European</td>
      <td>Italian</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>New York</td>
      <td>Chinese</td>
      <td>Dim Sum</td>
      <td>Italian</td>
      <td>Seafood</td>
      <td>Shop</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>Paris</td>
      <td>French</td>
      <td>Middle Eastern</td>
      <td>Cafeteria</td>
      <td>Szechuan</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>Rome</td>
      <td>Italian</td>
      <td>Sushi</td>
      <td>Chinese</td>
      <td>Pizza</td>
      <td>Indonesian</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>Seoul</td>
      <td>BBQ</td>
      <td>Korean</td>
      <td>Indian</td>
      <td>Asian</td>
      <td>Italian</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>Sydney</td>
      <td>Japanese</td>
      <td>Chinese</td>
      <td>Italian</td>
      <td>Australian</td>
      <td>French</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>Tokyo</td>
      <td>French</td>
      <td>Café</td>
      <td>Japanese</td>ho
      <td>Diner</td>
    </tr>
  </tbody>
</table>
</div>

Now the clusters can be plotted on a world map, with each cluster represnted with a unique colour, using the python libray Folium.
```python
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

world_map = folium.Map(location=[30, 30], zoom_start = 1.5)

# set color scheme for the clusters
x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(data['Lat'], data['Long'], top_cuisines["City"], top_cuisines["Cluster Labels"]):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(world_map)
       
world_map.save("world_map.html")
```
[Map of Clustered Cities](/images/clustered_cities_map.html)

### Discussion

We can see that Delhi and Hong Kong, Rome and Lisbon have their own clusters, reflecting their unique cuisine landscape. Thus these locations would be most suited to travelers desiring a unique culinary experience. Tokyo and Paris are strangely clustered together, however when I researched each city it is apparent that the restaurants around Tokyo city centre are heavily influenced by French cuisine. The remaining cities are clustered together meaning they are more similar to each other than any other city, thus meaning they have a less diverse range of cuisines than the other cities.

### Conclusion

This project was able to successfully highlight some commonalities between the major restaurant cuisines in cities around the world. It should be noted that better results may be obtained by extending the radius of the Foursquare search, increasing the number of cities and also increasing the number of restaurants used in the data. However to keep computational times low and to avoid exceeding the maximum number of requests to the Foursquare API these numbers were kept low.




