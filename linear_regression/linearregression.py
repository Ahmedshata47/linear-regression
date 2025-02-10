import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




df=pd.read_csv("Your path")
X=df.drop("Target",axis=1)
y=df["Target"]



def plot(w1,w2,b):
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x1_range = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 50)
    x2_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    y_pred_grid = w1 * X1_grid + w2 * X2_grid + b  

    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, label="Actual Data", c='blue', marker='o')

    ax.plot_surface(X1_grid, X2_grid, y_pred_grid, color='red',
                    alpha=0.5,label=f"w1 = {w1:.2f}\nw2 = {w2:.2f}\n b = {b:.2f}")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target")
    ax.set_title("3D Regression Plane for Linear Regression")

    ax.legend()
    plt.draw()
    plt.show()




class bgd:
    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None  
        self.bias = 0  

    def fit(self,X,y):

        self.weights=np.zeros(X.shape[1])


        for _ in range(self.epochs):

            y_pred=np.dot(X,self.weights)+self.bias

            dm=(-2/X.shape[0])*np.dot(X.T,(y-y_pred))
            db=(-2/X.shape[0])*np.sum((y-y_pred))

            self.weights=self.weights-self.learning_rate*dm
            self.bias=self.bias-self.learning_rate*db



    def predict(self,X):

        return np.dot(X, self.weights) + self.bias




model=bgd()
model.fit(X,y)
plot(model.weights[0],model.weights[1],model.bias)




