import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io.wavfile import read
import os
plt.style.use('dark_background')
fig,(ax,ax2) = plt.subplots(nrows = 2,figsize = (300,60))
a = read(f'{os.getcwd()}\\BWV 578.wav')
Fs = a[0] #medidas por segundo
dt = 1/Fs #tiempo entre dos medidas

N = 1024  # VARIABLE potencia de 2 numero de medidas en un bloque, se lo puede modificar si se quiere cambiar la presicion en tiempo y la presicion de frecuenccia Dt*df = 1 incertidumbre

Dt = N*dt #longitud del bloque
df = Fs/N #frecuencia mas chica analizable y diferencia entre frecuencias analisadas

def sigmoid(x):
  return (2/(1 + np.e**(-(x-treshold)/(3*treshold)))-1)
def color_temp(temp):
	temp = temp*66
	if temp <= 66:
		r = 255
	else:
		r = temp - 60
		r = 329.698727446 * (r ** -0.1332047592)
		r = min(255, max(0, r))

	if temp < 66:
		g = temp
		g = 99.4708025861 * np.log(g) - 161.1195681661
		g = min(255, max(0, g))
	else:
		g = temp - 60
		g = 288.1221695283 * (g ** -0.0755148492)
		g = min(255, max(0, g))

	if temp >= 65:
		b = 255
	elif temp < 20:
		b = 0
	else:
		b = temp - 10
		b = 138.5177312231 * np.log(b) - 305.0447927307
		b = min(255, max(0, b))

	return (r/255, g/255, b/255)
n = np.arange(int(N/2))
# si la onda tiene frecuencia (Fs - f) me queda igual a una con f por eso queda simetrica la fft por eso solo escribo hasta la mitad

freq_spectrum = n/Dt 
x = []
y = []
#guardo en un array todas las muestras de la onda
for pair in a[1]:
  x.append(pair[0])
x = np.array(x)
num_points = len(x)
num_blocks = int(num_points/N)
F = []

treshold = 300000

#calculo la fft para cada bloque y la guardo en un array los numeros se guardan si son mayores a un treshold y se guardan como sigmoidea del valor absoluto de cada elemento del array de numeros complejos, los que no superan el treshold se guardan como False y no se grafican
for k in range(num_blocks):
  block_wave = x[k*N:(k+1)*N]
  f = fft(block_wave)
  h = sigmoid(abs(f))*1*(abs(f) > treshold)
  F.append(h)
  if k % 1000 == 0:
    print('generando f:',k)

start = 4 #como no puedo graficarlo todo en una sola figura lo divido en 5 partes y las grafico separadas, la primera es la 0 y la ultima la 4

#grafico un ractangulo para cada valor de frecuencia y cada bloque si la fft en ese punto supera el treshold con un color determinado por las funciones sigmoid y color
for k in range(int(num_blocks/5)):
  k = k + start*int(num_blocks/5)
  f = F[k]
  if k % 100 == 0:
    print('graficando f:',k)
  for j in range(int(N/16)):
    t0 = k*Dt - start*int(num_blocks/5)*Dt
    f0 = j*df
    val = f[j]
    if val:
      #descomentar para hacer el grafico logaritmico ax.plot((t0,t0+Dt/2,t0+Dt/2,t0,t0),(np.log(f0)/np.log(2**(1/12)),np.log(f0)/np.log(2**(1/12)),np.log(f0)/np.log(2**(1/12))+1,np.log(f0)/np.log(2**(1/12))+1,np.log(f0)/np.log(2**(1/12))),'-',linewidth = 4,color = np.roll(color_temp(val),1))
      ax.plot((t0,t0+Dt/2,t0+Dt/2,t0,t0),(f0,f0,f0+df/2,f0+df/2,f0),'-',linewidth = 4,color = color_temp(val))

#grafico la parte de la onda correspondiente al espectrograma
t = np.arange(0,num_points*dt,dt)
ax2.plot(t[0:int(num_points/5)],x[start*int(num_points/5):(start + 1)*int(num_points/5)],'.')

#grafico puntos para fijar los ejes al principio y al final de la curva
ax.plot(num_points/5*dt,60)
ax2.plot(num_points/5*dt,60)
ax.plot(0,60)
ax2.plot(0,60)
