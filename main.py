import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# regr = MLPRegressor(
#    hidden_layer_sizes=(2), #sim 2
#    max_iter=8000,
#    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
#    solver='adam',
#    learning_rate='adaptive',
#    n_iter_no_change=600
# )

regr1 = MLPRegressor(hidden_layer_sizes=(4,2),max_iter=12000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=600)

regr2 = MLPRegressor(hidden_layer_sizes=(50,40,30),max_iter=16000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=800)

regr3 = MLPRegressor(hidden_layer_sizes=(70,60,50,40,30,20),max_iter=22000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=1000)

regr = regr2

print('Carregando Arquivo de teste')

arquivo = np.load('./data/teste4.npy')

x = arquivo[0]
y = np.ravel(arquivo[1])

media = 0
erros = []

for i in range(10):
    # print('Treinando RNA')
    regr = regr.fit(x,y)
    
    erros.append(regr.best_loss_)
    media += regr.best_loss_

    # print('Preditor')
    y_est = regr.predict(x)

    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)

    #plt.show()
    plt.savefig(f'./imagens/teste4/sim1/E{i+1}.png')

media = media / 10

variancia = 0

for i in erros:
    variancia += math.pow(i-media, 2)

desvio = math.sqrt(variancia/10)

print("media: ", end=" ")
print(media)
print("desvio: ", end=" ")
print(desvio)
