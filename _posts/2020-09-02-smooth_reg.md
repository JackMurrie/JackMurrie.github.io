---
title: "Smooth Regression and General Additive Models"
date: 2020-08-27
tags: [R , Statistical Methods]
header:
  image: "/images/michael-dziedzic-nbW-kaz2BlE-unsplash.jpg"
excerpt: "Text Classification using Model Stacking on Big Data"
mathjax: "true"
---

### Data
This data set contains 111 measurements of ozone concentration, radiation, temperature and wind. 

```R
air <- read.table("/Data/air.txt", header = TRUE)
attach(air)
head(air)
```

![](/images/perceptron/air_data.png)

### Smoothing Splines

$$RSS(f, \lambda)$$