import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


files_train = glob.glob('train/*.jpg')
files_test = glob.glob('test/*.jpg')
n_test = len(files_test)


imagenes_train = []
imagenes_validation = []

for i in range(100):
    imagen = image.imread('train/'+str(i+1)+'.jpg')
    imagenes_train.append(imagen)
    
for i in range(n_test):
    imagen = image.imread(files_test[i])
    imagenes_validation.append(imagen)

imagenes_train = np.array(imagenes_train,dtype=float)
imagenes_train = np.mean(imagenes_train,axis=3)
imagenes_validation = np.array(imagenes_validation,dtype=float)
imagenes_validation = np.mean(imagenes_validation,axis=3)


y_train = 1-np.arange(1,101)%2
x_train = imagenes_train.reshape((100, -1))[:,2000:3024]

x_validation = imagenes_validation.reshape((10, -1))[:,2000:3024]


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)

ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]


proyeccion_train = np.matmul(x_train,vectores)[:,:30]
proyeccion_test = np.matmul(x_test,vectores)[:,:30]
proyeccion_validation = np.matmul(x_validation,vectores)[:,:30]

hiperparametros = np.logspace(-3,2,20)
scores = []
for C in hiperparametros:
    clasificador = SVC(C=C)
    clasificador.fit(proyeccion_train,y_train)
    predicciones = clasificador.predict(proyeccion_test)
    scores.append(metrics.f1_score(y_test,predicciones,average='macro'))
    
mejor_C = hiperparametros[np.argmax(scores)]
print(np.amax(scores))

plt.figure()
plt.plot(hiperparametros,scores)
plt.xscale('log')
plt.scatter(hiperparametros[np.argmax(scores)],np.amax(scores))
plt.savefig('SVC')


clasificador = SVC(C=mejor_C)
clasificador.fit(proyeccion_train,y_train)
predicciones = clasificador.predict(proyeccion_validation)


out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, predicciones):
    print(f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))

out.close()