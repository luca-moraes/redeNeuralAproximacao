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

regr1_sim1 = MLPRegressor(hidden_layer_sizes=(50,40),max_iter=12000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=600)
#media:  0.3029508388863308
#desvio:  0.01890619579751386
regr1_sim2 = MLPRegressor(hidden_layer_sizes=(50,40,30),max_iter=12000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=600)
#media:  0.28139403108203775
#desvio:  0.004863126700328806
regr1_sim3 = MLPRegressor(hidden_layer_sizes=(70,60),max_iter=12000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=600)
#media:  0.2892289751444392
#desvio:  0.010376975968887255


regr2_sim1 = MLPRegressor(hidden_layer_sizes=(50,40,30),max_iter=16000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=800)
#media:  5.471932747561779
#desvio:  0.09947904174134116
regr2_sim2 = MLPRegressor(hidden_layer_sizes=(50,40),max_iter=16000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=800)
#media:  6.58950778697848
#desvio:  2.60148529753164
regr2_sim3 = MLPRegressor(hidden_layer_sizes=(60,50),max_iter=16000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=800)
#media:  5.6320480546623735
#desvio:  0.37222792753303663

regr5_sim1 = MLPRegressor(hidden_layer_sizes=(100,80,70,40,20,10),max_iter=522000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=22000)

regr5_sim2 = MLPRegressor(hidden_layer_sizes=(80,70,60,20),max_iter=822000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=32000)

regr5_sim3 = MLPRegressor(hidden_layer_sizes=(120,60),max_iter=322000,activation='relu',solver='adam',learning_rate='adaptive',n_iter_no_change=22000)

regr = regr5_sim3

print('Carregando Arquivo de teste')

arquivo = np.load('./data/teste5.npy')

x = arquivo[0]
y = np.ravel(arquivo[1])

media = 0
erros = []

for i in range(1):
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
    plt.savefig(f'./imagens/teste5/sim3/E{i+1}.png')

media = media / 10

variancia = 0

for i in erros:
    variancia += math.pow(i-media, 2)

desvio = math.sqrt(variancia/10)

print("media: ", end=" ")
print(media)
print("desvio: ", end=" ")
print(desvio)
