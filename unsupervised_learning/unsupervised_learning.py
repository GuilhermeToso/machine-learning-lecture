import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import pickle

np.random.seed(0)

class Hyperparameters():

    def __init__(self) -> None:
        super().__init__()

        self.wine_color = "#892E4A"
        self.beer_color = "#D98E04"
        self.whisky_color = "#A63F03"

        self.wine_color_mean = 500
        self.wine_color_std = 8
        self.wine_alcohol_mean = 14
        self.wine_alcohol_std = 2
        self.wine_class = 0
    
        self.beer_color_mean = 440
        self.beer_color_std = 13.33
        self.beer_alcohol_mean = 7
        self.beer_alcohol_std = 1
        self.beer_class = 1

        self.whisky_color_mean = 430
        self.whisky_color_std = 10.
        self.whisky_alcohol_mean = 46
        self.whisky_alcohol_std = 2.67
        self.whisky_class = 2


        self.test_size = 0.2

class Data(Hyperparameters):

    def __init__(self) -> None:
        super().__init__()
        self.df = None

    def generate(self):
        
        wine_color_samples = np.random.normal(self.wine_color_mean, self.wine_color_std,size=(333,1))
        wine_alcohol_samples = np.random.normal(self.wine_alcohol_mean, self.wine_alcohol_std,size=(333,1))
        wine_samples = np.hstack((wine_color_samples,wine_alcohol_samples))
        
        beer_color_samples = np.random.normal(self.beer_color_mean, self.beer_color_std,size=(333,1))
        beer_alcohol_samples = np.random.normal(self.beer_alcohol_mean, self.beer_alcohol_std,size=(333,1))
        beer_samples = np.hstack((beer_color_samples,beer_alcohol_samples))
        
        whisky_color_samples = np.random.normal(self.whisky_color_mean, self.whisky_color_std,size=(334,1))
        whisky_alcohol_samples = np.random.normal(self.whisky_alcohol_mean, self.whisky_alcohol_std,size=(334,1))
        whisky_samples = np.hstack((whisky_color_samples,whisky_alcohol_samples))
        
        samples = np.vstack((wine_samples, beer_samples, whisky_samples))

        self.df = pd.DataFrame(samples, columns=["Cor","Alcool"])
        self.df = self.df.sample(frac=1).reset_index(drop=True)


class ULFramework():

    def __init__(self) -> None:
        self.hp = Hyperparameters()
        self.data = Data()
        self.data.generate()
        self.model = self.learn()

    def plot_data(self):

        
        fig, ax = plt.subplots()
        x_1, x_2 = self.data.df.Cor.to_numpy(), self.data.df.Alcool.to_numpy()
        ax.scatter(x_1, x_2, c="blue", label="Desconhecido", edgecolors='black')
        ax.legend(title="Classe")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Cor (nm)", fontsize=20)
        plt.ylabel("Álcool (%)", fontsize=20)
        plt.show()

    def plot_clustered_data(self):
        _, inverse_index = np.unique(self.prediction(), return_inverse=True)

        colors = np.array([[217,142,4,204],[137,46,74,204],[166,63,3,204]],dtype=np.float32)
        colors /= 255.

        y_colors = colors[inverse_index]

        plt.scatter(
            self.data.df.Cor.to_numpy(), 
            self.data.df.Alcool.to_numpy(), c=y_colors, s=50)

        centers = self.model.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Cor (nm)", fontsize=20)
        plt.ylabel("Álcool (%)", fontsize=20)
        plt.show()

    def plot_distribution(self):

        # Extract x and y
        x = self.data.df.Cor.to_numpy()
        y = self.data.df.Alcool.to_numpy()

        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.set_xlabel("Cor (nm)", fontsize=20)
        ax.set_ylabel("Álcool (%)", fontsize=20)
        plt.show()

    def learn(self):
        X = self.data.df.to_numpy()
        kmeans_model = KMeans(n_clusters=3).fit(X)
        return kmeans_model

    def save_ul_model(self):

        with open("./unsupervised_learning/ul_model/model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_ul_model(self):

        with open("./unsupervised_learning/ul_model/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def prediction(self):
        y_means = self.model.predict(self.data.df.to_numpy())
        return y_means

    
