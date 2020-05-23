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

Firstly one-hot-encoding was performed on the city venue data and then grouped the data by city to obtain the mean of each cuisine within that city. The top 5 cuisines within each city. 

