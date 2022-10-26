#!/usr/bin/env python
# coding: utf-8

# # Modelo de Ingreso-Gasto Keynesiano

# Sebastian Torres - 20201586

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# - Parte informe 
# 
# Lectura: Dancourt, O. (2017). The lean times in the Peruvian economy. Journal of Post Keynesian Economics, 40(1), 112-129.

# ¿Cuál fue el impacto del choque externo adverso sufrido durante el periodo de 2014-2015 en la economía peruana? Partiendo de dicha pregunta, Dancourt describe el impacto recesivo e inflacionario de dicho choque y sus consecuencias en nuestra economía.  Posteriormente, busca analizar las políticas monetarias y fiscales con las que las autoridades respondieron desde el Banco Central de Reserva del Perú (BCRP) y el Ministerio de Economía (MEF). Por un lado, explica que el BCRP comprometió la política monetaria futura al reducir a la mitad su reserva de divisas (superó sus ventas de dólares) y mostró un mal manejo de la tasa de interés referencial al optar por una respuesta moderada en lugar de un ciclo agresivo de reducción (lo que habría tenido un mayor efecto positivo). Por otro lado, respecto a la política fiscal, argumenta que el paquete de reducción de impuestos fue inadecuado, debiendo haberse optado por un aumento en la inversión pública. 
# 
# Resulta útil resaltar dos fortalezas de la investigación. En primer lugar, el autor emplea una metodología centrada en patrones históricos: analiza los periodos de la economía a partir de una división entre épocas de vacas gordas (auges) y flacas (recesiones). Por ende, a pesar de la ausencia de algún método matemático, la constante comparación entre dichos períodos permite comprender el contenido y las críticas planteadas. En segundo lugar, es destacable la presencia de recomendaciones macroeconómicas planteadas por el autor. A diferencia de la gran mayoría de autores en ciencias sociales, Dancourt no se queda en el análisis de una relación explicativa y/o una crítica, sino que adicionalmente ofrece una lista de medidas alternativas para solucionar los retos pendientes para el próximo gobierno. 
# 
# Por otro lado, es necesario identificar dos debilidades. En primer lugar, no realiza una explicación respecto a la presencia de factores internacionales y el origen del choque externo o del periodo de “vacas flacas”. Si bien la presencia de estos sucesos en la economía peruana no es un caso aislado -han habido diversos casos durante los últimos 15 años-, sigue siendo necesaria una explicación causal entre el contexto mundial y el inicio de crisis o recesiones. Siguiendo al Fondo Monetario Internacional, es importante considerar la relación del Perú como una economía pequeña, pequeña y altamente sensible a choques externos ocurridos en China y EEUU desde el 2012 (2014b). En segundo lugar, sería recomendable contrastar los hallazgos agregando teorías críticas o perspectivas contrarias a los supuestos keynesianos empleados. De esta forma, la hipótesis del artículo e incluso las recomendaciones vertidas por el autor se fortalecerán mediante la refutación de posibles contrafácticos.  
# En línea con lo mencionado, el principal aporte de Dancourt es demostrar una relación entre el crecimiento económico y los precios de los commodities: una reducción de estos precios contribuye a generar recesiones. Asimismo, identifica que la desaceleración económica se debió principalmente al desplome de los precios mundiales de los metales, la caída de la inversión pública y privada y la sequía crediticia en dólares. 
# 
# Finalmente, considero que existen dos principales próximos pasos para avanzar en esta pregunta. En primer lugar, la exploración de vías de enfoques económicos nacionales alternativos. Como propone Lust, el Estado necesita impulsar programas de diversificación de nuestra economía capitalista de subsistencia para así reducir el nivel de vulnerabilidad frente a las recesiones internacionales, especialmente tomando en cuenta la variable internacional de interdependencia asimétrica (2018: 97-98). En segundo lugar, resultaría interesante extender este estudio a un número mayor de casos en el ámbito latinoamericano. Siguiendo a Mukherjee, la posición estructural de la mayoría de estos países como exportadores de materias primas es útil para encontrar similitudes y plantear patrones de respuesta (2016: 23). De esta manera, podemos encontrar soluciones óptimas para la situación de las economías periféricas latinoamericanas y mejorar la capacidad de respuesta frente a futuras crisis. Dichos avances cobran importancia en pleno contexto de recesión económica por factores nacionales (crisis política) e internacional (pandemia). 
# 
# 
# 
# Bibliografía:
# 
# Lust, J. (2018). Vista del surgimiento de una economía capitalista de subsistencia en el Perú.
# http://revistas.urp.edu.pe/index.php/pluriversidad/article/view/1673/1529. 
# 
# Mukherjee, D. (2016). Informal economy in emerging economies: not a substitute but a complement!. International Journal of Business and Economic Development (IJBED), 4(3).
# 
# https://www.bolpress.com/2018/12/10/el-surgimiento-de-una-economia-capitalista-de-subsistencia-en-el-peru/
# 

# Parte código

# Pregunta 1: 

# Sabiendo que: 

# $$ C = C_0 + bY^d $$

# , y que:  

# $$ T = tY $$ 
# 
# $$ Y^d = Y - T -> Y^d = (1-t)Y $$

#  - Donde, tributación total (T) = tY, donde "t" es la tasa de impuestos e "Y" es el ingreso agregado

# Entonces, parte de este ingreso disponible Y^d se utiliza para los gastos en consumo. De aquí, se obtiene el consumo variable como parte de la función de demanda de Consumo. Así: 

# $$ C = C_0 + bY^d $$

# Considerando el diferencial de la función de demanda de Consumo del ejercicio 1:

# $$ \frac{∆C}{∆Y^d} = \frac{(C_1-C_0)}{(Y^d_1 - Y^d_0)}  $$
# 

# $$ \frac{(40-30)}{(40-20)} = \frac{10}{20} = 0.5 $$

# Explicando la función de demanda de Consumo:
# $$ C = C_0 + bY^d $$
# 
# Donde: 
#     
#     C_0  = Consumo autónomo (consumo fijo).
#     
#     bY^d = Consumo variable, donde: 
#         
#         Y^d = Ingreso disponible. Representa el ingreso restante luego del que el Estado cobre la tasa impositiva.
#     
#         b = Propensión marginal a consumir. Es el cambio en el consumo por unidad de cambio en el ingreso disponible, ∈
#         (]0,1[).

# Entonces:
# 
# La función de consumo representa que, en general, todas las familias del país se ven ante una situación de doble forma de consumo. Hay consumo autónimo (fijo) y consumo variable (variación del consumo según el ingreso disponible).  

# Graficando: 

# In[3]:


# Parámetros

Y_size = 100 

Co = 35
b = 0.8
t = 0.3

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def C(Co, b, t, Y):
    C = Co + b*(1-t)*Y
    return C

C = C(Co, b, t, Y)
# Gráfico

# Dimensiones del gráfico
y_max = np.max(Y)
fig, ax = plt.subplots(figsize=(10, 8))

custom_xlim = (0, 130)
custom_ylim = (20, 130)

plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)


# Curvas a graficar
ax.plot(Y, C, label = "Consumo", color = "#3D59AB") #Demanda agregada

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto agregado
    # punto de equilibrio
plt.text(80, 95, '$PMgC = △C/△Y^d = b = 0.5$', fontsize = 13, color = 'black')


# Título y leyenda
ax.set(title="Función de demanda de Consumo", xlabel= '$Y^d$', ylabel= 'C')
ax.legend() #mostrar leyenda

plt.show()


# Pregunta 2: 

# Considerando el diferencial de la función de demanda de Inversión:

# $$ ∆I = ∆I_0 - ∆hr $$

# Sabiendo que: 
# 
#     - Los componentes autónomos no cambian = ∆I_0 = 0

# $$ ∆I = - ∆hr $$

# $$ ∆I = (-) $$

# $$ ∆I < 0 $$

# Explicando la función de demanda de Inversión:
# $$ I = I_0 - hr $$
# 

# Donde: 
#     
#     I = Inversión autónoma (inversión fija).
#     r = Tasa de interés real. Es la tasa nominal neta de la inflación esperada.  
#     h = Sensibilidad de los inversionistas ante la variación de la taza de interés (relación negativa), ∈ (]0,1[) y dato
#         fijo.  

# Entonces:
# 
# Si la inversión autonóma se incrementa, en principio, se incrementaría también la inversión en el país. Sin embargo, si la taza de interés sube, la inversión decaerá.    

# Graficando: 

# In[4]:


z_size = 100          
z = np.arange(z_size) 
z                     


# In[5]:


def funcion_2(x, z):
    funcion_2 = (x - z)
    return funcion_2       


# In[6]:


funcion_2(100, z)


# In[7]:


f2 = funcion_2(100, z)


# In[8]:


Z = np.arange(z_size)


# In[9]:


a = -2.5 

def L_45(a, Z):
    L_45 = a*Z
    return L_45

L_45 = L_45(a, Z)


# In[10]:


y_max = np.max(L_45)
fig, ax = plt.subplots(figsize=(8, 6))
   
ax.plot(Z, L_45, color = "#404040") 

ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

plt.text(80, -180, '$I = I_0 - hr$', fontsize = 11.5, color = 'black')

ax.set(title="Función de demanda de Inversión", xlabel= r'r', ylabel= r'I')


# Pregunta 3:

# - El modelo supone que el nivel de precios está fijo.
# - El nivel del producto se adapta a los cambios en la demanda agregada. 
# - La tasa de interés está determinada fuera del modelo (en el mercado monetario). 
# - Es un modelo de corto plazo. 

# Pregunta 4: 

# Sabiendo que: 

# $$ C = C_0 + bY^d $$
# $$ I = I_0 - hr $$
# $$ G = G_0 $$
# $$ T = tY $$
# $$ X = X_0 $$
# $$ M = mY^d $$

# $$ DA = C + I + G + X -M $$

# Podemos reemplazando la función con los datos previos:

# $$ DA = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$
# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + Y(b - m)(1 - t) $$
# 

# Entonces, $ DA = α_0 + α_1Y $
# * Intercepto: $ α_0 = C_0 + I_0 + G_0 + X_0 - hr $ 
# * Pendiente: $ α_1 = (b - m)(1 - t) $
# 

# Reemplazamos en la ecuación del Ingreso de equilibrio: 
# - En donde  $Y = DA$
# 

# $$ Y = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# $$ k=\frac{1}{1 - (b - m)(1 - t)} $$
# 

# Entonces:

# In[11]:


# Parámetros

Y_size = 100 

Co = 36
Io = 41
Go = 71
Xo = 3
h = 0.6
b = 0.7 # b > m
m = 0.1
t = 0.2
r = 0.8

Y = np.arange(Y_size)


# In[12]:


def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)



# In[13]:


a = 2.5

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# Graficamos:

# In[14]:


# Dimensiones del gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(15, 13))

# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #Demanda agregada
ax.plot(Y, L_45, color = "#404040") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=74.5,  ymin= 0, ymax= 0.73, linestyle = ":", color = "green")
plt.axhline(y=186, xmin= 0, xmax= 0.8, linestyle = ":", color = "green")
plt.axhline(y=150, xmin= 0, xmax= 0.8, linestyle = ":", color = "green")

# Texto agregado
    # punto de equilibrio
plt.text(0, 188, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 152, '$α_o$', fontsize = 15, color = 'black')
plt.text(78, 180, '$α_1 = (b-m)(1-t)Y$', fontsize = 15, color = 'black')
plt.text(78, 155, '$α_0 = (C_0 + I_0 + G_0 + X_0 - hr)$', fontsize = 15, color = 'black')
plt.text(90, 215, '$DA = α_0 + α_1Y$', fontsize = 15, color = 'black')
plt.text(92, 205, '$↓$', fontsize = 18, color = 'grey')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'red')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a CP", xlabel= r'Y', ylabel= r'DA')
plt.show()


# Pregunta 5:

#     a) Política fiscal Expansiva con aumento del Gasto del Gobierno (∆G>0)
#     
#         - Análisis intuitivo: 

# $$ ↑Go → ↑DA → DA > Y → ↑Y $$

# Interpretación: Con una política Fiscal Expansiva, el gobierno aumenta el gasto público para expandir la producción
# y el empleo. Sabemos que las variables exógenas (tasa de interés real, el consumo, la inversión y las exportaciones 
# autónomas) no cambian y se presuponen constantes la presión tributaria y las propensiones marginales a consumir
# e importar. Por ende, el aumento en el Gasto de Gobierno hará que se eleve la Demanda Agregada (DA), de manera que
# la nueva DA será mayor que la demanda agregada inicial, razón por la cual será necesario aumentar el nivel de
# producción para volver al punto de equilibrio. 

#     b) Análisis matemático: ∆Go > 0 → ¿∆Y?

# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# Pero, si no ha habido cambios en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$

# Sabiendo que $∆G_0 > 0 $ y que $k > 0$, la multiplicación de un número positivo con un positivo dará otro positivo:
# 
# $$ ∆Y = (+)(+) $$
# $$ ∆Y > 0 $$

#     c) Análisis gráfico: 

# In[15]:


# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 95

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[16]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 82,  ymin= 0, ymax = 0.80, linestyle = ":", color = "grey")
plt.axhline(y = 206, xmin= 0, xmax = 0.80, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 125, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 180, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(85, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(75, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(15, 160, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Aumento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# b) Política fiscal Expansiva con reducción de la tasa de tributación (∆t<0)
# 
#         - Análisis intuitivo: 

# $$ t↓ → Co↑ → DA↑ → DA > Y → Y↑ ... (1) $$
# $$ t↓ → M↑ → DA↓ → DA < Y → Y↓ ... (2) $$

# Interpretación: Por un lado, en (1), si la tasa de tributación disminuye, y por ende los impuestos que las personas tienen que pagar disminuye, el nivel de consumo probablemente aumente (relación inversa) en tanto las personas podrán tener un ingreso disponible mayor y, por lo tanto, la demanda de consumo y consecuentemente la demanda agregada suban. Entonces, la nueva demanda agregada será mayor que el nivel de producción inicial (las personas van a demandar más de lo que se está ofertando) y, por ende, para regresar al equilibrio se tendrá que aumentar el nibel de producción. 
#     Por otro lado, en (2), si la tasa de tributación disminuye, también ocurre que las exportaciones netas de importaciones disminuirían (relación directa), razón por la cual la demanda agregada caerá. En este escenario, la nueva demanda agregada será menor que el nivel de producción (la economía produce más de lo que la gente demanda). Entonces, será necesario reducir el nivel de producción para regresar al nivel de equilibrio.
# 

# $ OJO $ :

# Entonces, tomando en cuenta (1) y (2), parece haber una contradiccion. Por el lado del consumo, la reducción de la tasa impositiva provoca que el nivel de producción aumente. Pero al mismo tiempo, por el lado de las importaciones y exportaciones, la reducción de la tasa impositiva provoca que el nivel de producción caiga. La única forma de solucionar este dilema es con el análisis matemático a continuación. 

#     - Matemáticamente: ∆t < 0  →  ¿∆Y?

# In[17]:


Co, Io, Go, Xo, h, r, b, m, t = symbols('Co Io Go Xo h r b m t')

f = (Co + Io + Go + Xo - h*r)/(1-(b-m)*(1-t))


df_t = diff(f, t)
df_t #∆Y/∆t


# Considernado el diferencial de $∆t$:
# 
# $$ \frac{∆Y}{∆t} = \frac{(m-b)(Co + Go + Io + Xo - hr)}{(1-(1-t)(b-m)+1)^2} $$
# 
# - Sabiendo que b > m, entonces $(m-b) < 0$
# - Los componentes autónomos no cambian: $∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$
# - Cualquier número elevado al cuadrado será positivo: $ (1-(1-t)(b-m)+1)^2 > 0 $
# 
# Entonces:
# 
# $$ \frac{∆Y}{∆t} = \frac{(-)}{(+)} $$
# 
# Sabemos que $∆t < 0$ y que la multiplicación de dos negativos da un positivo, el cual al ser dividido con otro positivo da positivo:
# 
# $$ \frac{∆Y}{(-)} = \frac{(-)}{(+)} $$
# 
# $$ ∆Y = \frac{(-)(-)}{(+)} $$
# 
# $$ ∆Y > 0 $$

# Entonces, comprobamos que el impacto principal de una reducción en la tasa impositiva se da en el consumo. Esto dado que en la economía de un país es el consumo agregado de las familias el componente con mayor peso respecto a las importaciones. Por ende, el impacto de la tasa impositiva siempre será considerado por el camino del consumo autónomo de las familias (siempre será mayor). 

#     - Análisis gráfico: 

# In[18]:


# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3 #tasa de tributación
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
t = 0.02

# Generar la ecuación con el nuevo parámetros
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_t = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[19]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_t, label = "DA_t", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 77,  ymin= 0, ymax = 0.75, linestyle = ":", color = "grey")
plt.axhline(y = 192, xmin= 0, xmax = 0.75, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 200, '$DA_t$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(80, 0, '$Y_t$', fontsize = 12, color = '#EE7600')
plt.text(72, 45, '$→$', fontsize = 18, color = 'red')
plt.text(20, 180, '$↑$', fontsize = 18, color = 'red')

# Título y leyenda
ax.set(title = "Reducción de la Tasa de Tributación", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# Pregunta 6:

#    a) 6.1

# In[20]:


Y_size = 100 

Co = 36
Io = 41
Go = 71
Xo = 3
h = 0.6
b = 0.7 
m = 0.1
t = 0.2
r = 0.8
g = 0.6

Y = np.arange(Y_size)


# In[21]:


def DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t) - g)*Y
    return DA_K # Ecuación de la curva del ingreso de equilibrio

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y)



# In[22]:


a = 2.5

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y) # Recta de 45°


# In[23]:


y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(15, 13))

ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #Demanda agregada
ax.plot(Y, L_45, color = "#404040") #Línea de 45º

ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=71,  ymin= 0, ymax= 0.70, linestyle = ":", color = "green")
plt.axhline(y=178, xmin= 0, xmax= 0.8, linestyle = ":", color = "green")
plt.axhline(y=142, xmin= 0, xmax= 0.8, linestyle = ":", color = "green")


# Texto agregado
    # punto de equilibrio
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 152, '$α_o$', fontsize = 15, color = 'black')
plt.text(80, 170, '$α_1 = (b-m)(1-t)Y$', fontsize = 15, color = 'black')
plt.text(78, 150, '$α_0 = (C_0 + I_0 + G_0 + X_0 - hr)$', fontsize = 15, color = 'black')
plt.text(90, 120, '$DA = α_0 + α_1Y$', fontsize = 15, color = 'black')
plt.text(92, 125, '$↑$', fontsize = 18, color = 'grey')

    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')

ax.set(title="El ingreso de Equilibrio a CP", xlabel= r'Y', ylabel= r'DA')
plt.show()



#     b) 6.2

# $ Entonces: $

# A partir de la condición Y = DA, y resolviendo en función de Y, se obtiene la ecuación   del   ingreso   de   equilibrio así: 

# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + Y((b - m)(1 - t)- g) $$

# Ahora, considerando la condición de equilibrio $Y = DA$, la ecuación del ingreso de equilibrio a corto plazo es:

# $$ Y = \frac{1}{1 - (b - m)(1 - t) + g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 

#     c) 6.3

#     - Análisis gráfico: 

# In[24]:


# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 36
Io = 41
Go = 71
Xo = 3
h = 0.6
b = 0.7 
m = 0.1
t = 0.2
r = 0.8
g = 0.6

Y = np.arange(Y_size)

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t) - g)*Y
    return DA_K 


DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y)



#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 90

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t) - g)*Y
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[25]:


y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 65, ymin= 0, ymax = 0.64, linestyle = ":", color = "grey")
plt.axhline(y = 160, xmin= 0, xmax = 0.63, linestyle = ":", color = "grey")
plt.axvline(x = 58,  ymin= 0, ymax = 0.58, linestyle = ":", color = "grey")
plt.axhline(y = 143, xmin= 0, xmax = 0.57, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 140, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 162, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(56, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(65, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(60, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 150, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = " Aumento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


#     - Análisis intuitivo:

# $$ ↑Go → ↑DA → DA > Y → ↑Y $$

# Interpretación: El gobierno decide aumentar su gasto autónomo cuando la economía entra en recesión y/o en depresión. En este contexto, disminuye la producción y aumenta el desempleo de la fuerza laboral.  Ante ellos, el gobierno incrementa su gasto para estimular la recuperación de la producción. En este sentido,  si la economía entra en la fase de expansión y auge, el producto y el empleo aumentan

#     - Análisis matemático:  

# - Matemáticamente: $∆G_0 > 0  →  ¿∆Y?$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t) + g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# Pero, si no ha habido cambios en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$
# 

# Sabiendo que $∆G_0 > 0 $ y que $k > 0$, la multiplicación de un número positivo con un positivo dará otro positivo:
# 
# $$ ∆Y = (+)(+) $$
# $$ ∆Y > 0 $$

#     - Parte 4: 

# Como se puede observar en la parte 3, el parámetro "g" de la regla contracíclica aparece en dos ecuaciones: 

# $$ Y = \frac{1}{1 - (b - m)(1 - t) + g} (C_0 + I_0 + G_0 + X_0 - hr)$$ 

# - Entonces, en la ecuación el parámetro "g" aparece aumentando la magnitud del denominador del multiplicador y, por tanto, reduce su tamaño (del multiplicador keynesiano). Por otro lado, en la segunda ecuación, el parámetro disminuye la magnitud de la pendiente de la función de DA. 

#     - Parte 6.5

# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + Y((b - m)(1 - t)- g) $$

# Respuesta: No, como se evidencia en la ecuación, el parámetro "g" disminuye la magnitud de la pendiente de la función de DA. Este tipo de política fiscal hace que los auges y las depresiones no sean muy pronunciados. Se establece como una regla consistente en una variación del Gasto (G) en sentido contrario a la variación del Producto. Entonces, la política fiscal de gasto contra cíclico afecta la pendiente de la curva de DA y el valor del multiplicador. Ambos disminuyen y, por ende, morigera las fluctuaciones del producto o ingreso. 

#     - Parte 6.6:

# Análisis gráfico

# In[26]:


# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 36
Io = 41
Go = 71
Xo = 6
h = 0.6
b = 0.7 
m = 0.1
t = 0.2
r = 0.8
g = 0.6

Y = np.arange(Y_size)

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t) - g)*Y
    return DA_K 


DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y)



#--------------------------------------------------
# NUEVA curva 

    # Definir SOLO el parámetro cambiado
Xo = 1

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t) - g)*Y
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[27]:


y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 58, ymin= 0, ymax = 0.58, linestyle = ":", color = "grey")
plt.axhline(y = 148, xmin= 0, xmax = 0.60, linestyle = ":", color = "grey")
plt.axvline(x = 56,  ymin= 0, ymax = 0.56, linestyle = ":", color = "grey")
plt.axhline(y = 140, xmin= 0, xmax = 0.57, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 162, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 140, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(60, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(54, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(55, 45, '$←$', fontsize = 18, color = 'grey')
plt.text(20, 140, '$↓$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = " Reducción de exportaciones $(X_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# Análisis intuitivo

# $$ ↓Xo → ↓DA → DA < Y → ↓Y $$

# Interpretación: Las exportaciones decaen, por lo cual la demanda baja ante la escacez de productos. En este sentido, el nivel de producción inicial (antes de que las exportaciones cayeran) será mayor a la demanda agregada actual. En este escenario, se deberá reducir el nivel de producción para volver al equilibrio. 

# Análisis matemático

# - Matemáticamente: $∆G_0 < 0  →  ¿∆Y?$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t) + g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# Pero, si no ha habido cambios en $C_0$, $I_0$, $G_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆G_0 = ∆h0 = ∆r = 0$$
# 
# $$ ∆Y = k (∆X_0) $$

# Sabiendo que $∆X_0 < 0 $ y que $k > 0$, la multiplicación de un número negativo con un positivo dará otro negativo:
# 
# $$ ∆Y = (-)(+) $$
# $$ ∆Y < 0 $$
