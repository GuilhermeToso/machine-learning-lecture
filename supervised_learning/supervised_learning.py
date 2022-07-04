import numpy as np
from scipy import rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.epochs = 10
        

class Data(Hyperparameters):

    def __init__(self) -> None:
        super().__init__()
        self.df = None

    def generate(self):
        
        wine_color_samples = np.random.normal(self.wine_color_mean, self.wine_color_std,size=(333,1))
        wine_alcohol_samples = np.random.normal(self.wine_alcohol_mean, self.wine_alcohol_std,size=(333,1))
        wine_class_samples = np.zeros(shape=(333,1)) + self.wine_class
        wine_samples = np.hstack((wine_color_samples,wine_alcohol_samples,wine_class_samples))
        
        beer_color_samples = np.random.normal(self.beer_color_mean, self.beer_color_std,size=(333,1))
        beer_alcohol_samples = np.random.normal(self.beer_alcohol_mean, self.beer_alcohol_std,size=(333,1))
        beer_class_samples = np.zeros(shape=(333,1)) + self.beer_class
        beer_samples = np.hstack((beer_color_samples,beer_alcohol_samples,beer_class_samples))
        
        whisky_color_samples = np.random.normal(self.whisky_color_mean, self.whisky_color_std,size=(334,1))
        whisky_alcohol_samples = np.random.normal(self.whisky_alcohol_mean, self.whisky_alcohol_std,size=(334,1))
        whisky_class_samples = np.zeros(shape=(334,1)) + self.whisky_class
        whisky_samples = np.hstack((whisky_color_samples,whisky_alcohol_samples,whisky_class_samples))
        
        samples = np.vstack((wine_samples, beer_samples, whisky_samples))

        self.df = pd.DataFrame(samples, columns=["Cor","Alcool", "Classe"])
        self.df = self.df.replace({'Classe':{0:"Vinho",1:"Cerveja",2:"Whisky"}})
        self.df = self.df.sample(frac=1).reset_index(drop=True)


class SLFramework():

    def __init__(self) -> None:
        self.hp = Hyperparameters()
        self.data = Data()
        self.data.generate()
        self.scaler = StandardScaler()
        self.preprocess_data()
        self.model = self.make_model()

    def scale_data(self, x):

        return self.scaler.fit_transform(x)


    def split_data(self, x, y):

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.hp.test_size, random_state=0
        )

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    
    def preprocess_data(self):

        y = self.data.df["Classe"].replace({"Vinho":0,"Cerveja":1,"Whisky":2}).to_numpy()
        x = self.data.df.drop(["Classe"], axis=1).to_numpy()

        x = self.scale_data(x)

        self.split_data(x,y)
    
    def make_model(self):

        model = Sequential()
        model.add(Dense(32, activation="relu", name="dense_layer_1"))
        model.add(Dense(16, activation="relu", name="dense_layer_2"))
        model.add(Dense(8, activation="relu", name="dense_layer_3"))
        model.add(Dense(3, activation="sigmoid", name="output_layer"))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def learn(self):

        self.preprocess_data()
        self.model.fit(
            self.x_train, self.y_train, epochs=self.hp.epochs
        )

    def save_sl_model(self):
        self.model.save("./supervised_learning/sl_model")

    def load_sl_model(self):

        self.model = load_model("./supervised_learning/sl_model")

    def evaluate_model(self):
        return self.model.evaluate(self.x_test,self.y_test)

    def prediction(self, x):

        if x.ndim == 2:
            x = self.scaler.transform(x)
        elif x.ndim == 1:
            x = self.scaler.transform([x])
        return self.model.predict(x)


    def plot_data(self):

        y = self.data.df["Classe"].replace({"Vinho":0,"Cerveja":1,"Whisky":2}).to_numpy()
   
        fig, ax = plt.subplots()
        labels = ["Vinho", "Cerveja", 'Whisky']
        color = [self.hp.wine_color, self.hp.beer_color, self.hp.whisky_color]
        for i in range(3):
            x_1, x_2 = self.data.df.Cor.to_numpy(), self.data.df.Alcool.to_numpy()
            row_idx = np.where(y == i)
            x_1, x_2 = x_1[row_idx], x_2[row_idx]
            ax.scatter(x_1, x_2, c=color[i], label=labels[i], edgecolors='black')

        ax.legend(title="Classe")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Cor (nm)", fontsize=20)
        plt.ylabel("Álcool (%)", fontsize=20)
        plt.show()


    def plot_decision_boundary(self, data="train"):

        if (data == "train"):
            x, y = self.x_train, self.y_train
        elif (data == "test"):
            x, y = self.x_test, self.y_test
        
        x = self.scaler.inverse_transform(x)

        _, inverse_index = np.unique(y, return_inverse=True)

        colors = np.array([[137,46,74,204], [217,142,4,204],[166,63,3,204]],dtype=np.float32)
        colors /= 255.

        y_colors = colors[inverse_index]

        min_cor, max_cor = self.data.df.Cor.min()-1, self.data.df.Cor.max()+1
        min_alcool, max_alcool = self.data.df.Alcool.min()-1, self.data.df.Alcool.max()+1

        cor_grid = np.arange(min_cor, max_cor, 0.1)
        alcool_grid = np.arange(min_alcool, max_alcool,0.1)

        xx, yy = np.meshgrid(cor_grid, alcool_grid)

        vec1, vec2 = xx.flatten(), yy.flatten()
        vec1, vec2 = vec1.reshape((len(vec1),1)), vec2.reshape((len(vec2),1))
    
        grid = np.hstack((vec1, vec2))

        y_predict = self.prediction(grid)

        y_predict = y_predict.argmax(axis=-1)

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
