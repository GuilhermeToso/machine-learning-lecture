import numpy as np
import sklearn
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pickle

np.random.seed(0)

class Hyperparameters():

    def __init__(self) -> None:
        super().__init__()

        self.wine_color = "#C909DE"
        self.beer_color = "#F5F116"
        self.whisky_color = "#F56C00"

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
        self.train = None
        self.test = None

    def generate(self):
        unlabelIndex = int(0.8*267)

        wine_color_train_samples = np.random.normal(self.wine_color_mean, self.wine_color_std,size=(267,1))
        wine_alcohol_train_samples = np.random.normal(self.wine_alcohol_mean, self.wine_alcohol_std,size=(267,1))
        wine_class_train_samples = np.zeros(shape=(267,1)) + self.wine_class
        wine_class_train_samples[:unlabelIndex] = -1
        wine_train_samples = np.hstack((wine_color_train_samples,wine_alcohol_train_samples,wine_class_train_samples))

        wine_color_test_samples = np.random.normal(self.wine_color_mean, self.wine_color_std,size=(66,1))
        wine_alcohol_test_samples = np.random.normal(self.wine_alcohol_mean, self.wine_alcohol_std,size=(66,1))
        wine_class_test_samples = np.zeros(shape=(66,1)) + self.wine_class
        wine_test_samples = np.hstack((wine_color_test_samples,wine_alcohol_test_samples,wine_class_test_samples))
        
        beer_color_train_samples = np.random.normal(self.beer_color_mean, self.beer_color_std,size=(267,1))
        beer_alcohol_train_samples = np.random.normal(self.beer_alcohol_mean, self.beer_alcohol_std,size=(267,1))
        beer_class_train_samples = np.zeros(shape=(267,1)) + self.beer_class
        beer_class_train_samples[:unlabelIndex] = -1
        beer_train_samples = np.hstack((beer_color_train_samples,beer_alcohol_train_samples,beer_class_train_samples))

        beer_color_test_samples = np.random.normal(self.beer_color_mean, self.beer_color_std,size=(66,1))
        beer_alcohol_test_samples = np.random.normal(self.beer_alcohol_mean, self.beer_alcohol_std,size=(66,1))
        beer_class_test_samples = np.zeros(shape=(66,1)) + self.beer_class

        beer_test_samples = np.hstack((beer_color_test_samples,beer_alcohol_test_samples,beer_class_test_samples))
        
        whisky_color_train_samples = np.random.normal(self.whisky_color_mean, self.whisky_color_std,size=(267,1))
        whisky_alcohol_train_samples = np.random.normal(self.whisky_alcohol_mean, self.whisky_alcohol_std,size=(267,1))
        whisky_class_train_samples = np.zeros(shape=(267,1)) + self.whisky_class
        whisky_class_train_samples[:unlabelIndex] = -1
        whisky_train_samples = np.hstack((whisky_color_train_samples,whisky_alcohol_train_samples,whisky_class_train_samples))

        whisky_color_test_samples = np.random.normal(self.whisky_color_mean, self.whisky_color_std,size=(66,1))
        whisky_alcohol_test_samples = np.random.normal(self.whisky_alcohol_mean, self.whisky_alcohol_std,size=(66,1))
        whisky_class_test_samples = np.zeros(shape=(66,1)) + self.whisky_class

        whisky_test_samples = np.hstack((whisky_color_test_samples,whisky_alcohol_test_samples,whisky_class_test_samples))
        
        train_samples = np.vstack((wine_train_samples, beer_train_samples, whisky_train_samples))
        test_samples = np.vstack((wine_test_samples, beer_test_samples, whisky_test_samples))

        self.train = pd.DataFrame(train_samples, columns=["Cor","Alcool", "Classe"])
        self.train = self.train.sample(frac=1).reset_index(drop=True)

        self.test = pd.DataFrame(test_samples, columns=["Cor","Alcool", "Classe"])
        self.test = self.test.sample(frac=1).reset_index(drop=True)


class SSLFramework():

    def __init__(self) -> None:
        self.hp = Hyperparameters()
        self.data = Data()
        self.data.generate()

    def learn(self):

        x = self.data.train[["Cor","Alcool"]].to_numpy()
        y = self.data.train.Classe.to_numpy()
        self.model = LabelPropagation().fit(x,y)


    def save_ssl_model(self):

        with open("./semisupervised_learning/ssl_model/model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_ssl_model(self):

        with open("./semisupervised_learning/ssl_model/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, x):
        y_means = self.model.predict(x)
        return y_means

    def evaluate(self):

        yhat = self.prediction(self.data.test[["Cor","Alcool"]].to_numpy())
        score = accuracy_score(self.data.test.Classe.to_numpy(), yhat)
        print('Accuracy: %.3f' % (score*100))


    def plot_train_data(self):

        y = self.data.train.Classe.to_numpy()
        fig, ax = plt.subplots()
        labels = ["Desconhecido","Vinho", "Cerveja", 'Whisky']
        color = ["black",self.hp.wine_color, self.hp.beer_color, self.hp.whisky_color]
        for i in range(4):
            feature_1, feature_2 = self.data.train.Cor.to_numpy(), self.data.train.Alcool.to_numpy()
            row_idx = np.where(y == i-1)
            feature_1, feature_2 = feature_1[row_idx], feature_2[row_idx]
            ax.scatter(feature_1, feature_2, c=color[i], label=labels[i], edgecolors='black')

        ax.legend(title="Classe")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Cor (nm)", fontsize=20)
        plt.ylabel("Álcool (%)", fontsize=20)
        plt.show()

    def plot_decision_boundary(self, data="train"):

        if (data == "train"):
            x, y = self.data.train[["Cor","Alcool"]].to_numpy(),self.prediction(self.data.train[["Cor","Alcool"]].to_numpy())
        elif (data == "test"):
            x, y = self.data.test[["Cor","Alcool"]].to_numpy(),self.prediction(self.data.test[["Cor","Alcool"]].to_numpy())
        
        
        
        _, inverse_index = np.unique(y, return_inverse=True)

        colors = np.array([[201, 9, 222,204], [245, 241, 22,204],[166,63,3,204]],dtype=np.float32)
        colors /= 255.

        y_colors = colors[inverse_index]

        min_cor, max_cor = x[:,0].min()-1, x[:,0].max()+1
        min_alcool, max_alcool = x[:,1].min()-1, x[:,1].max()+1

        cor_grid = np.arange(min_cor, max_cor, 0.1)
        alcool_grid = np.arange(min_alcool, max_alcool,0.1)

        xx, yy = np.meshgrid(cor_grid, alcool_grid)
        print(xx.shape)
        vec1, vec2 = xx.flatten(), yy.flatten()
        vec1, vec2 = vec1.reshape((len(vec1),1)), vec2.reshape((len(vec2),1))
    
        grid = np.hstack((vec1, vec2))
        
        y_predict = self.prediction(grid)
        
        zz = y_predict.reshape(xx.shape)

        contour = plt.contourf(xx,yy,zz,levels=[0,1,2],extend="min",colors=colors)
        cb = plt.colorbar(contour)
        cb.set_ticklabels(["Vinho","Cerveja","Whisky"])
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)

        for class_value in range(3):
            row_ix = np.where(y == class_value)
            plt.scatter(x[row_ix, 0], x[row_ix, 1], linewidth=1.5, edgecolors="#000",
                        c=y_colors[row_ix])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel("Cor (nm)", fontsize=20)
        plt.ylabel("Álcool (%)", fontsize=20)
        plt.show()


    

