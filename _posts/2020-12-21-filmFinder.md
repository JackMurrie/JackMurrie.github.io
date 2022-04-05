---
title: "UNSW CSE Capstone Project"
date: 2020-12-21
tags: [frontend development, docker]
excerpt: "A web application to browse, track and get recommendations for movies"
mathjax: "true"
---

Four of my classmates and myself created a web application for our computer science capstone course. The application allows users to signup, browse, search, add movies to a seen list and 
a wishlist, get personal recommendations, view and block other users, leave ratings and reviews and play a film finding game in which mutiple
users connect and vote for their favourite movie. 

We had a great team and the project was enjoyable to work on. My role was a frontend developer, using the javascript library React and material-UI components. 

[FilmFinder Source Code](https://github.com/JackMurrie/filmfinder)

To start the app locally:

```
git clone git@github.com:JackMurrie/filmfinder.git
```
```
cd filmfinder
```
```
docker-compose up -d
```

This command builds both the backend and frontend development images and runs the containers. 
The backend is served from `http://localhost:8080/`, while the frontend can be accessed at `http://localhost:3000/`