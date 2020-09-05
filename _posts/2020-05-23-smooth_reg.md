---
title: "Smooth Regression and General Additive Models"
date: 2020-05-23
tags: [R , Statistical Methods]
excerpt: "Smoothing spline regression and model fitting on Ozone Concentration data in R"
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

In order to model non-linear relationships we must fit smooth functions. In the case of smoothing splines, we find a function f that minimizes the penalized residual sum of sqaures:

$$RSS(f, \lambda)=\sum_{i=1}^{n}(y_i-f(x_i))^2+\lambda \int(f^"(t))^2dt$$

Here <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> is a smoothing parameter, that establishes a tradeoff between the first term - the closeness or goodness of fit of the data (variance) and the second term - the level of curvature of <img src="https://latex.codecogs.com/gif.latex?f" title="f" /> (bias).

This can be rewritten as:

$$RSS(\bf{\theta},\lambda)=||\bf{y}-N\theta||^2+\lambda\theta^T\int N_j^"(t) N_k^"(t) dt$$

Where <img src="https://latex.codecogs.com/gif.latex?N" title="N" /> is a matrix of K dimensional basis function sets for representing the appropriate family of natural splines.

The solution is:
$$\hat{\theta} = (N^TN+\lambda\int N_j^"(t) N_k^"(t) dt)^{-1}N^Ty$$

And the fitted smoothing spline is:

$$\hat{r}_n(x)=\sum_{j=1}^{n}\hat{\theta}_jN_j(x)$$

We can fit smoothing splines to each covariate:
```R
plot(radiation, ozone)
#cubic smoothing spline
lines(smooth.spline(radiation, ozone), col = "red")
```
![](/images/perceptron/rad_oz.png)
```R
plot(temperature, ozone)
#cubic smoothing spline
lines(smooth.spline(temperature, ozone), col = "blue")
```
![](/images/perceptron/temp_oz.png)
```R
plot(wind, ozone)
#cubic smoothing spline
lines(smooth.spline(wind, ozone), col = "green")
```
![](/images/perceptron/wind_oz.png)

### Density Estimation

R uses Kernal Density Estimation to estimate densities for non-paramteric data:
$$\hat{f}(x)=\frac{1}{nh}\sum_{i=1}^{n}K(\frac{x-x_i}{h})$$

Where <img src="https://latex.codecogs.com/gif.latex?h" title="h" /> is a tuning paramter, the bandwidth and <img src="https://latex.codecogs.com/gif.latex?K(u)" title="K(u)" /> is a kernal function.

We can compare our density estimate with the actual data.
```R
hist(radiation, prob=TRUE)
lines(density(radiation), col="red")
```
![](/images/perceptron/hist_rad.png)

### Model Fitting

General additive models are a form of local regression in which mutivariate data can be fiited with non-linear relationships to the response.

Suppose that <img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /> has a distribution from the exponential family:

$$f(y_i;\theta_i,\phi)=exp(A_i(y_i\theta_i-c(\theta_i))/\phi+h(y_i,\phi/A_i)) $$

In a GAM we assume that the link function 
$$g(\mu_i)=\sum_{j=1}^{p}f_j(x_{ij})$$
where the <img src="https://latex.codecogs.com/gif.latex?f" title="f" />'s are a collection of smooth univariate functions where the responses are indepentant and <img src="https://latex.codecogs.com/gif.latex?x_{ij}" title="x_{ij}" />'s are the p predictor variables for observation i.

We minimise, <img src="https://latex.codecogs.com/gif.latex?l" title="l" /> is log-likelihood:

$$-2\phi l(f)+\sum_{j=1}^{p}\lambda_j \int(f_j^"(x_j))^2dx_j$$

Of which the solution is a smoothing spline. In R the above is calculated via the fisher scoring algorithm - a version of the backfitting algorithm is used a each scoring step.

```R
library(mgcv)
air.gam <- gam(ozone ~ s(radiation) + s(temperature) + s(wind), data = air)
summary(air.gam)
```
![](/images/perceptron/gam_out.png)
As can be seen above the predictors all account for significant influence in the response.
```R
par(mfrow = c(2, 2))
plot(air.gam)
```
![](/images/perceptron/gam_plots.png)
We can also plot interactions.
```R
air.gam <- gam(ozone ~ s(temperature) + s(radiation), data = air)
grid <- list(temperature = seq(from = 57, to = 97, length = 50),
radiation = seq(from = 7, to = 334, length = 50))
air.pr <- predict(air.gam, newdata = expand.grid(grid))
air.pr.mat <- matrix(air.pr, ncol = 50, nrow = 50)
persp(grid$temperature, grid$radiation, air.pr.mat,
xlab = "temperature", ylab = "radiation", zlab = "ozone",
theta = -45, phi = 15, d = 2.0, tick = "detailed")
```
![](/images/perceptron/gam_int.png)