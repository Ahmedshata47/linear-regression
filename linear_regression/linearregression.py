import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#-------------------------Loaiding_the_data-------------------------#
df = pd.read_csv("Your path")
x1=df["Feature"]
y1=df["Target"]



#-------------------------Ploting_the_fitted_line-------------------------#

colors = ['red', 'blue', 'green', 'orange', 'purple']
color_index = 0

def plot(m,b):

    global color_index

    plt.scatter(x=x1, y=y1, alpha=0.5, color="blue")
    x_range = np.linspace(0, 10, 1000)  
    line, = plt.plot(x1, m * x1 + b, color=colors[color_index], linewidth=5)
    color_index = (color_index + 1) % len(colors)  


    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.pause(0.01)
    line.remove()
    

def gradient_descent(m,b,df,lr):

    dm=0
    db=0
    n=len(df)

    
    dm += -(2/n) *np.dot(x1.T,y1-(m*x1+b))
    db += -(2/n) * np.sum(y1-(m*x1+b))

    m=m-lr*dm
    b=b-lr*db

    return m,b

m=0
b=0
lr=0.01
ephocs=1000

for _ in range(ephocs):
    m,b=gradient_descent(m,b,df,lr)
    plot(m,b)
