---
title: "Simulating Brownian Motion"
date: 2020-05-21
tags: [statistics, modelling]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Simulating Brownian Motion and using it to model stock prices"
mathjax: "true"
---
### Simulating Brownian Motion

Brownian Motion is the seemingly random movement of particles suspended in a fluid (i.e. dust motes in water or air) and by extension the mathematical model used to describe this movement. It is one of the simplest and most fundamental continuous-time
stochastic processes, finding applications in numerous situations.

A stochastic process $X(t), t \geq 0$ is Brownian if: \
$X(0) = 0$ (arbitrary choice)\
$X(t), t \geq 0$ has stationary and independant increments\
and for every $t > 0, X(t) \sim \mathcal{N}(0, \sigma^2t)$

Thus it follows, for any $t, s > 0$:\

$$X(s+t)|X(s)=x) \sim \mathcal{N}(x,\,\sigma^{2}t)$$

When $\sigma = 1$ the process is known as Standard Brownian Motion 


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


![png]("/BM_outputs/output_3_0.png")


