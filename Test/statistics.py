import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats

###############################
# Ex1
auto_turismo = 1000713
auto_industriales = 144558
auto_agrícolas = 40584
moto = 63791
ciclomotores = 210462
ciclos = 581271

data = np.array([1000713, 144558, 40584, 63791, 210462, 581271])
mean = np.mean(data)
mean = "{:.1f}".format(mean)
media = np.median(data)
media = "{:.1f}".format(media)
mode = stats.mode(data)
mode = mode.mode[0]
mode = "{:.1f}".format(mode)
fig, axs = plt.subplots(nrows = 2, ncols=1, figsize = (16,8))
x = ['Automóviles de turismo', 'Vehíuclos industriales', 'Vehículos agrícolas',
     'Motociclestas y motocarros', 'Ciclomotores', 'Ciclos']

axs[0].stem(x, data)
axs[0].set_ylabel('Unidades vendidas')
axs[0].set_title('Bienes Manufacturados 1977')
axs[1].pie(data, labels=x)
axs[1].annotate('Media  ' + str(mean), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(1.8, 0.25), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Mediana  ' + str(media), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(1.8, 0.16), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Moda  ' + str(mode), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(1.8, 0.05), textcoords='axes fraction', va='top', ha='left')

plt.show()

###############################
#Ex 3
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['Nº hijos 0','Nº hijos 1', 'Nº hijos 2', 'Nº hijos 3', 'Nº hijos 4', 'Nº hijos 5', 'Nº hijos 6',
          'Nº hijos 7', 'Nº hijos 8', 'Nº hijos 9']
familias = np.array([21, 18, 26, 17, 9, 4, 3, 1, 0, 1])

lista_media = 21*[0]+18*[1]+26*[2]+17*[3]+9*[4]+4*[5]++3*[6]+1*[7]+0*[8]+1*[9]
n = len(lista_media)
if (n % 2) == 0:

    # print(“Thenumber is even”)
    index = int((n)/2)

else:
    # print(“The providednumber is odd”)
    index = int((n-1)/2)


mean = np.sum(x*familias)/np.sum(familias)
#
media = lista_media[index]
#
mode = stats.mode(lista_media)
mode = mode.mode[0]
std = np.std(lista_media)
CV = (np.std(lista_media)/mean)*100
mean = "{:.1f}".format(mean)
media = "{:.1f}".format(media)
mode = "{:.1f}".format(mode)
CV = "{:.1f}".format(CV)



fig, axs = plt.subplots(nrows = 2, ncols=1, figsize = (16,8))

axs[0].stem(x, familias)
axs[0].set_ylabel('Nº de familias')
axs[0].set_xlabel('Nº de hijos')
axs[0].set_title('Ayudas familiares')
axs[1].pie(familias, labels=labels)
axs[1].annotate('Media  ' + str(mean), xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.25), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Mediana  ' + str(media), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(1.8, 0.16), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Moda  ' + str(mode), xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.05), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Coeficiente de Variación  ' + str(CV)+" %", xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.33), textcoords='axes fraction', va='top', ha='left')

plt.show()

###############################
#Ex 3
operarios = [((10-7.5)/2)+7.5]*15 + [((13-10)/2)+10]*25 + [((16-13)/2)+13]*9 + [((19-16)/2)+16]*21 + \
            [((22-19)/2)+19]*28 + [((25-22)/2)+22]*21
operarios_items = [15, 25, 9, 21, 28, 12]
operarios_labels = ["7-10", "10-13", "13-16", "16-19", "19-22", "22-25"]
bins_edges = [7, 10, 13, 16, 19, 22, 25]
labels = ['Nº hijos 0','Nº hijos 1', 'Nº hijos 2', 'Nº hijos 3', 'Nº hijos 4', 'Nº hijos 5', 'Nº hijos 6',
           'Nº hijos 7', 'Nº hijos 8', 'Nº hijos 9']
sum_items = np.sum(operarios_items)
fr = operarios_items/sum_items
cumulative_items = np.cumsum(operarios_items)
# computing mean
mid_points = []
for i, item in enumerate(bins_edges):
    if i < len(bins_edges)-1:
        marca_point = (bins_edges[i+1]+bins_edges[i])/2
        mid_points.append(marca_point)

mid_points = np.array(mid_points)
mean = np.sum(mid_points*fr)
test_mean = np.mean(operarios)
# computing media
# find where
if (sum_items % 2) == 0:
#
#     # print(“Thenumber is even”)
     index = int((sum_items)/2)
#
else:
#     # print(“The providednumber is odd”)
     index = int((sum_items-1)/2)

index_where = np.where(cumulative_items < index)
index_find = np.max(index_where)+1
media = bins_edges[index_find] + ((index - cumulative_items[index_find-1])/operarios_items[index_find])*(bins_edges[index_find+1]-bins_edges[index_find])

# computing moda
moda_num = operarios_items[index_find+1] - operarios_items[index_find]
moda_den = (operarios_items[index_find+1] - operarios_items[index_find]) + (operarios_items[index_find+1] - operarios_items[index_find+2])
mode = bins_edges[index_find] + (bins_edges[index_find+1]-bins_edges[index_find])*(moda_num/moda_den)


# computing the standard deviation
# computing CV


std = np.sqrt(np.sum(operarios_items*((mid_points-mean)**2))/sum_items)
CV = std/mean
mean = "{:.2f}".format(mean)
media = "{:.2f}".format(media)
mode = "{:.2f}".format(mode)
CV = "{:.2f}".format(CV)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (16,8))
axs[0].hist(operarios, bins=bins_edges, linewidth=0.5, edgecolor="white")

axs[0].set_xlabel('Salario anual, miles de euros')
axs[0].set_ylabel('Número de operarios')
axs[0].set_title('Ingresos Anuales')
axs[1].pie(operarios_items, labels=operarios_labels)
axs[1].annotate('Media  ' + str(mean), xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.25), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Mediana  ' + str(media), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(1.8, 0.16), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Moda  ' + str(mode), xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.05), textcoords='axes fraction', va='top', ha='left')
axs[1].annotate('Coeficiente de Variación  ' + str(CV), xy=(0.1, 0.1), xycoords='axes fraction',
                     xytext=(1.8, 0.33), textcoords='axes fraction', va='top', ha='left')

plt.show()
