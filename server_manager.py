import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from multiprocessing import Process
from sklearn.metrics import mean_absolute_error
from multiprocessing.managers import BaseManager
from lib.feature.LinearExpression import LinearFe
from sklearn.model_selection import train_test_split
from controller.svmEventControl import control as svmControl
from controller.lineEventControl import control as lineControl
from controller.neighborsEventsControl import control as neighborsControl
from controller.polynomialEventControl import control as polynomialControl

class ServerManager(BaseManager):

    def __init__(self)-> None:
        self.processes = []
        self.df = pd.read_csv('dataSet/kc_house_data.csv')

    #################################################################################################
    ########### need to think about how to pass the score to customer after fit finished#############
    #################################################################################################

    def multiProcessStart(
            self, 
            server
        )-> None:
    
        control_functions = [lineControl, neighborsControl, polynomialControl, svmControl]
        for control_func in control_functions:
            self.processes.append(
                Process(
                    target=control_func,
                    args=(server, self)
                )
            )

        # Start all processes
        for process in self.processes: process.start()

        # Wait for all processes to complete
        for process in self.processes: process.join()


    def start(
                self, 
                server,
                key = None
            )-> None:


            if key == 'MULTI': self.multiProcessStart(server)
            elif key == 'LINE':lineControl(server, self)
            elif key == 'NEIGHBORS':neighborsControl(server, self)
            elif key == 'POLYNOMIAL':polynomialControl(server, self)
            elif key == 'SVM':svmControl(server, svmControl)
            else: self.shutdown()

    def shutdown(self) -> None:
        # terminate all processes
        for process in self.processes:
            process.terminate()
        # delete all processes
        self.processes = []
        return super().shutdown()
    
    def getDataSet(self) -> pd.DataFrame:
        return self.df
    
    def gerateTrainData(
            self,
            dataSet,
            x_dataDrop_key,
            y_dataDrop_key, 
            size, 
            random,
            key
        ) -> dict:

        x_train, x_test, y_train, y_test = train_test_split(
            dataSet.drop(x_dataDrop_key, axis=1), 
            dataSet[y_dataDrop_key], 
            test_size=size, 
            random_state=random
        )

        preditionData = {}

        if key == 'train':
            preditionData['x'] = x_train
            preditionData['y'] = y_train
        else:
            preditionData['x'] = x_test
            preditionData['y'] = y_test

        return preditionData
    
    def showhead(self)->pd.DataFrame:
        return self.df.head()
    
    def showInfo(self)->None:
        return self.df.info()
    
    def showDescrible(self)->pd.DataFrame:
        return self.df.describe()
    
    def showMeanAbsoluteError(self, test, pred)->( float or np.ndarray ):
        return mean_absolute_error(test, pred)
    
    def showScore(self, x, y, **kwargs)->float:

        model = LinearFe(x, y, **kwargs)

        return model.score(x, y)
    
    def showPredict(self, x, y, pred, **kwargs)->np.ndarray:

        model = LinearFe(x, y, **kwargs)

        return model.predict(pred)
        
    def showSns(self, trainArray)->None:
        sns.pairplot(
            self.df[trainArray]
        )
        plt.show()

    def showFigure(self, key, dict)->None:

        dataSet = self.df.copy()
        dataSet.isna().sum()
        dataSet['year'] = dataSet['date'].str.slice(0, 4) 
        dataSet['month'] = dataSet['date'].str.slice(4, 6) 
        dataSet['day'] = dataSet['date'].str.slice(6, 8) 
        dataSet = dataSet.drop('date', axis=1)
        dataSet = dataSet.drop('id', axis=1)
        dataSet.head(3)
        figureData = dict[key]

        if figureData == None:
            return None
        
        return px.histogram(
            dataSet, 
            x=key, 
            title=figureData["title"], 
            labels=figureData["labels"]
        )

    def kmeansCalcuate(
            self,
            trainArray,
            max_iter = 300,
            n_init = 10,
            random_state = 0 
        )->None:

        wcss = []

        x = self.df.iloc[:, trainArray].values

        y_Kmeans = None

        for i in range(1, 11):
            kmeans = KMeans(
                n_clusters= i, 
                init = 'k-means++',
                max_iter= max_iter,
                n_init = n_init,
                random_state = random_state
            )

        y_Kmeans = kmeans.fit_predict(x) 

        kmeans.fit(x)

        wcss.append(kmeans.inertia_)

        plt.plot(range(1, 11), wcss)

        plt.title('The elbow method')

        plt.xlabel('number of cluster')

        plt.ylabel('WCSS')

        plt.show()

        plt.scatter(
            x[y_Kmeans == 0, 0], 
            x[y_Kmeans == 0, 1], 
            s = 100, 
            c = 'red', 
            label = 'Iris-setosa'
        )

        plt.scatter(
            x[y_Kmeans == 1, 0], 
            x[y_Kmeans == 1, 1], 
            s = 100, 
            c = 'blue', 
            label = 'Iris-versicolour'
        )

        plt.scatter(
            x[y_Kmeans == 2, 0], 
            x[y_Kmeans == 2, 1], 
            s = 100, 
            c = 'green', 
            label = 'Iris-virginica'
        )

        plt.scatter(
            kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],  
            s = 100, 
            c = 'yellow', 
            label = 'Centroids'
        )

        plt.legend()

    




    





        