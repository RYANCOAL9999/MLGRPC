import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from multiprocessing import Process
from sklearn.metrics import mean_absolute_error
from multiprocessing.managers import BaseManager
from sklearn.model_selection import train_test_split
from controller.svmEventsControl import control as svmControl
from controller.lineEventsControl import control as lineControl
from controller.generalEventControl import control as generalControl
from controller.neighborsEventsControl import control as neighborsControl
from controller.polynomialEventsControl import control as polynomialControl

class ServerManager(BaseManager):

    def __init__(self, file_path: str)-> None:
        self.__processes = []
        self.__df = pd.read_csv(file_path)

    #################################################################################################
    ########### need to think about how to pass the score to customer after fit finished#############
    #################################################################################################

    def multiProcessStart(
            self, 
            server
        )-> None:
    
        control_functions = [lineControl, neighborsControl, polynomialControl, svmControl]
        with contextlib.ExitStack() as stack:
            for control_func in control_functions:
                self.__processes.append(
                    Process(
                        target=control_func,
                        args=(server, self)
                    )
                )
            stack.enter_context(process)
            self.__processes.append(process)

        # Start all processes
        for process in self.__processes: process.start()

        # Wait for all processes to complete
        for process in self.__processes: process.join()


    def start(
                self, 
                server,
                key = None
            )-> None:

            generalControl(server, self)

            control_functions = {
                'MULTI': self.multiProcessStart,
                'LINE': lineControl,
                'NEIGHBORS': neighborsControl,
                'POLYNOMIAL': polynomialControl,
                'SVM': svmControl
            }

            if key in control_functions:
                control_func = control_functions[key]
                control_func(server, self)
            else:
                self.shutdown()
    


    def shutdown(self) -> None:
        # terminate all processes
        for process in self.__processes:
            process.terminate()
        # delete all processes
        self.__processes = []
        return super().shutdown()
    
    def getDataSet(self) -> (pd.DataFrame or None):
        return self.__df if self.__df is not None else None
        
    def chooseDictData(
        self,
        key,
        x_train,
        y_train,
        x_test,
        y_test
    )-> dict:
        preditionData = {}
        if key == 'train':
            preditionData['x'] = x_train
            preditionData['y'] = y_train
        else:
            preditionData['x'] = x_test
            preditionData['y'] = y_test

        return preditionData
    
    def generateTrainData(
            self,
            dataSet,
            x_dataDrop_key,
            y_dataDrop_key, 
            size, 
            random
        ) -> list:

        return train_test_split(
            dataSet.drop(x_dataDrop_key, axis=1), 
            dataSet[y_dataDrop_key], 
            test_size=size, 
            random_state=random
        )
    
    def showhead(self)->pd.DataFrame:
        return self.__df.head()
    
    def showInfo(self)->None:
        return self.__df.info()
    
    def showDescrible(self)->pd.DataFrame:
        return self.__df.describe()
    
    def showMeanAbsoluteError(self, test, pred, **kwargs)->( float or np.ndarray ):

        return mean_absolute_error(test, pred, **kwargs)
    
    def showScore(self, x, y, model)->float:

        return model.score(x, y)
    
    def showPredict(self, model, pred)->np.ndarray:

        return model.predict(pred)
        
    def showSns(self, trainArray)->None:
        sns.pairplot(
            self.__df[trainArray]
        )
        plt.show()

    def showFigure(self, key, dict)->None:

        dataSetCopy = self.__df.copy()
        dataSetCopy.isna().sum()
        dataSetCopy['year'] = dataSetCopy['date'].str.slice(0, 4) 
        dataSetCopy['month'] = dataSetCopy['date'].str.slice(4, 6) 
        dataSetCopy['day'] = dataSetCopy['date'].str.slice(6, 8) 
        dataSetCopy = dataSetCopy.drop('date', axis=1)
        dataSetCopy = dataSetCopy.drop('id', axis=1)
        dataSetCopy.head(3)
        figureData = dict[key]

        if figureData == None:
            return None

        return px.histogram(
            dataSetCopy, 
            x=key, 
            title=figureData["title"], 
            labels=figureData["labels"]
        )

    def kmeansCalculate(
            self,
            trainArray,
            max_iter = 300,
            n_init = 10,
            random_state = 0 
        )->None:

        wcss = []

        x = self.__df.iloc[:, trainArray].values

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

    




    





        