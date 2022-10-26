#!/usr/bin/env python
# coding: utf-8

# # El mercado monetario y la curva LM

# Sebastian Torres - 20201586

# - Parte ensayo:
# 
# Mendoza, W., Mancilla, L., & Velarde, R. (2021). La Macroeconomía de la cuarentena: Un modelo de dos sectores. Pontificia Universidad Católica del Perú

# En este artículo, los autores proponen explicar los efectos que la cuarentena, como principal instrumento para contener la pandemia de COVID-19, tuvo en la economía peruana Para ello, construyen un modelo macroeconómico de dos sectores: el sector 1, compuesto por hoteles, restaurantes, aerolíneas, etc, como directamente afectado; y el sector 2, que engloba la producción de alimentos, bienes y servicios indispensables, como indirectamente afectado. Así, se analiza el impacto desigual de la cuarentena (contracción de la producción) en base a la naturaleza de los sectores. Finalmente, dado que dichos sectores se encuentran interconectados en base al consumo de los trabajadores, se estudian las políticas que impidieron el deterioro de la capacidad productiva de la economía en general. 
# 
# En cuanto a las fortalezas, el artículo destaca por su estructura. Los autores se preocuparon por mantener un orden cohesionado y una base teórica detallada para la compresión del modelo para, posteriormente, contextualizar el caso peruano. En primer lugar, se realiza la explicación de los dos sectores estudiados a través de la presentación de teoría macroeconómica (las funciones, conceptos y gráficos se explican a partir de los sectores). Ello permite una comprensión clara de lo acontecido durante el periodo 2020-2021 desde diversas posiciones -la cuarentena no afectó a todos de la misma manera. Por otra parte, una segunda fortaleza se encuentra en su metodología de análisis. En la segunda parte del trabajo, los autores derivan el modelo general en tres subsistemas para un estudio de ambos sectores en plazos: corto plazo, de equilibrio estacionario y de transición hacia el equilibrio estacionario. Dicha definición de plazos analíticos les permite a los autores profundizar más en los sistemas macroeconómicos por sector. 
# 
# Sin embargo, es posible identificar debilidades. Si bien su objetivo es poder explicar los impactos de la cuarentena específicamente en la economía nacional, tanto la pandemia como las respuestas políticas empleadas por parte del Estado responden a una serie de criterios diversos dentro de los cuales el aspecto económico es tan solo uno más. En otras palabras, la ausencia de factores o relaciones causales entre elementos sociales, políticos y económicos constituye un punto de crítica importante. Por ejemplo, si bien se asume que los agentes económicos proyectan sus expectativas solo a partir de la información sobre los precios en el periodo previo (sentido de racionalidad), en la realidad las personas se comportaron de manera impredecible frente al anuncio de la cuarentena. Se esperaba que las personas realicen un nivel de consumo proporcional y de gasto mesurado, pero en la práctica cada una buscó asegurar sus recursos para el futuro: comprando papel higiénico, alcohol y mascarillas en cantidades innecesarias.  En segundo lugar, el estudio se realiza sobre datos recolectados del sector económico formal, resultando entonces muy probablemente distante a la realidad dado el gran sector informal característico de la economía peruana.  
# 
# Aún con sus debilidades, el artículo representa un gran avance para el análisis de los impactos de la pandemia, desde el aspecto económico. Es destacable que, a pesar de tratarse de los pocos trabajos que han podido analizar el periodo reciente pero sumamente intrigante de la pandemia, se desarrolla el contenido de manera pedagógica y secuencial de manera que se pueda democratizar los hallazgos a un público más amplio que solamente economistas. Por último, incluye propuestas de solución para las dificultades encontradas al mismo tiempo que sugiere futuras investigaciones - como políticas públicas centradas en un eje combinado de salud pública y economía. 
# 
# Finalmente, considero que hay dos pasos fundamentales para avanzar en nuestra capacidad de comprensión de los impactos de la pandemia en la economía peruana. En primer lugar, es necesario incluir al análisis las afecciones en las fuentes de ingreso económico externo. Dada la posición estructural de Perú como una economía periférica centrada en la exportación de materias primas y recursos naturales, dichas exportaciones se posicionan como un factor principal en el PBI nacional. Como destaca Herska (2015), incluir el factor externo de nivel de exportaciones es importante para explicar el flujo de capitales y, por ende, la posibilidad de una recuperación económica más acelerada. Asimismo, la reducción de la fuerza laboral producto de la cuarentena tendría un impacto negativo dado que reduce la capacidad extractiva para la exportación. En segundo lugar, para el caso peruano se debe incluir necesariamente la variable política. Especialmente dado que gran parte de las dificultades para el desarrollo económico del país se originaron antes de la pandemia gracias a la gran inestabilidad política de los últimos años (Cavalié, Choy, Delgado et al., 2019). En este sentido, para entender las dificultades económicas y cómo se agravaron con la pandemia, debemos incluir factores políticos. 
# 
# 
# - Bibliografía: 
# 
# Cavalié, S., Choy, M., Delgado, L., & González, A. (2019). Factores que afectan la estabilidad política: un estudio econométrico.
# 
# Herska, C. (2015). Las relaciones económicas internacionales y las posibilidades de desarrollo del Perú. Apuntes: Revista de Ciencias Sociales, (11), 23-37.
# 

# ## Parte código:

# #### Integrantes: Sebastian Torres, Joaquin del Castillo

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
import numpy as np
import random
import math
import sklearn
import scipy as sp
import networkx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from causalgraphicalmodels import CausalGraphicalModel


# ### Pregunta 1: 

# #### 1.1) 
# 
# Son cuatro los instrumentos de la política monetaria:
# 
# 
# 1)Operación de Mercado Abierto: El BCR realiza estas operaciones comprando y vendiendo activos financieros o bonos a los bancos  comerciales. En el caso de una política monetaria expansiva se mediante la compra de bonos, con la cual el BCR inyecta soles en la economía. Mientras que en la política monetaria contractiva se da con la venta de bonos al mercado (bancos comerciales). Con esta operación, el Banco Central retira dinero (soles) de la economía. 
# 
# 
# 
# 2)El coeficiente legal de encaje como instrumento de política es el segundo instrumento. En el caso de una política moentaria expansiva el BCR puede aumentar la cantidad de dinero que poseen los bancos para generar préstamos mediante la reducción de la tasa de encaje, ello genera que los bancos comerciales tengan menos dinero en reservas en el balance del BCR. Ahora bien, en la política monetaria contractiva el BCR aumenta el coneficiente de encaje, los bancos comerciales tienen una mayor proporción de depósitos en reservas y ello reduce el multiplicador bancario y origina una disminución en la oferta monetaria. 
# 
# 
# 3)La tasa de interés como instrumento de política:A partir del 1990 la tasa de interés se transformó en instrumento de política y la oferta monetaria pasó a ser una variable endógena. La tasa de política es la tasa de interés de referencia de la política monetaria. En el caso de la política monetaria expansiva el BCR reduce la tasa de interés de referencia, con lo cual aumenta el dinero prestado a los bancos comerciales y ello incrementa la base monetaria y de esta forma la oferta monetaria. Mientras que en el caso de la política monetaria contractiva el BCR aumenta la tasa de interes de referencia, con lo cual reduce el dinero prestado a los bancos comerciales y ello reduce la base monetaria y de esta forma la oferta monetaria. 
# 
# 
# 4)Función de Oferta Real de Dinero: El autor supone que la oferta nominal de dinero (𝑀𝑠) es una variable exógena e instrumento de política monetaria. En términos reales o a precios constantes, la oferta nominal se divide entre en nivel general de precios de la economía, (𝑃). Se supone que el nivel de precios estan dados a corto plazo. 
# 

# #### 1.2) 
# 
# $${M^s_0}$$
# 
# Se puede ver, en primer lugar, a la masa monetaria, la cual es el dinero que crea el BCR y se encuentra en circulación en la economía. 
# 
# $$P$$
# 
# Se puede ver, en segundo lugar, al precio, el cual indica el valor real de cada billete o divisa. 
# 
# $$M^s= \frac{M^s_0}{P}$$
# 
# Finalmente, se puede ver que la división entre el número de billetes que transitan en la economía con el precio que valen, se obtiene a la oferta real de dinero. 

# #### 1.3)
# 
# 1)El motivo Transacción: Dada su función de medio de intercambio, el dinero se demanda para realizar transacciones. La magnitud de las transacciones de bienes y servicios está en relación directa con el ingreso o producto de la economía. Por ende, la demanda de dinero depende directamente del Ingreso (Y). 
# 
# 2)El motivo de Precaución:  En este caso, el dinero se demanda para pagar las deudas. Como los que se endeudan deben tener capacidad de pago, y esta depende de sus ingresos, a nivel macroeconómico la demanda de dinero por el motivo precaución también depende positivamente del ingreso (Y). 

# Entonces, en un primer momento, la demanda de dinero por motivos de transacción y precaución será: 
# 
# $$ L_1 = kY $$ 

# 3) El motivo Especulativo: El activo financiero dinero compite con el activo financiero no monetario bono en la función de reserva de valor. Se preferirá mantener liquidez en forma de dinero y no en forma de bonos cuando la tasa de interés se reduce y lo contrario si aumenta. Por ende, esta demanda dependerá inversamente de la tasa de interés de los bonos. 

# Entonces, en un segundo momento, la demanda de dinero por motivo de especulació será:

# $$ L_2 = -ji $$

# Por tanto, si asumimos que ambos tipos de demanda están en términos reales, la función de demanda real de dinero será: 

# $$ L = L_1 + L_2 $$
# $$ L = kY - ji $$
# 
# - Donde: 
#     - $ k $ = indica la sensibilidad de la demanda de dinero ante variaciones del ingreso (“Y”).
#     - $ j $ = indica la sensibilidad de la demanda de dinero ante las variaciones de la tasa de interés nominal de los bonos ("i"). 

# #### 1.4) 
# 
# El equilibrio en el Mercado de Dinero se deriva del equilibrio entre la Oferta de Dinero $(M^s)$ y Demanda de Dinero $(M^d)$:
# 
# $$ M^s = M^d $$
# 
# $$ \frac{M^s}{P} = kY - ji $$

# Dado que en el Corto Plazo (CP) el nivel de precios es fijo y exógeno, podemos asumir que la inflación esperada será 0. Por ende, no habría una gran diferencia entre la tasa de interés nominal $(i)$ y la real $(r)$.
# 
# Entonces la ecuación del equilibrio en el mercado monetario es:
# 
# $$ \frac{M^s}{P} = kY - jr $$
# 

# #### 1.5) 

# In[3]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[4]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_0$", fontsize = 12, color = 'black')
ax1.text(52, 0, "$(M_o^s) / P_0$", fontsize = 12, color = 'black')
ax1.text(85, 1, "$L(Y_0)$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Pregunta 2: 

# #### 2.1) ∆ Y < 0

# $ Explicación: $
# 
# Cuando disminuye el ingreso “ Y ”, la demanda de dinero también disminuye. Si el mercado estaba en equilibrio, esta disminución del ingreso genera un exceso de oferta en el mercado.  El equilibrio debe restablecerse con una disminución de la tasa de interés. La curva de demanda de dinero se desplaza hacia abajo.
# 
# $ Graficando: $

# In[6]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 50
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[7]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_1$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=15, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 15.4, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 15.3, "$E_0$", fontsize = 12, color = 'black')


plt.text(42, 12, '∆Y', fontsize=12, color='black')
plt.text(45, 11.9, '↓', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# #### 2.2) ∆ k < 0; si la elasticidad del ingreso disminuye:

# $ Explicación: $
# 
# Cuando disminuye la sensibilidad de la demanda de dinero ante variaciones del ingreso, la demanda de dinero también disminuye. Si el mercado estaba en equilibrio, esta disminución de la elasticidad del ingreso genera un exceso de oferta en el mercado (dado que a la economía le va mal, vamos a demandar menos dinero y, por ende, se reduce el nivel de consumo).  El equilibrio debe restablecerse con una disminución de la tasa de interés. La curva de demanda de dinero se desplaza hacia abajo.
# 
# $ Graficando: $

# In[8]:


# Parameters
r_size = 100

k = 0.8
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[9]:


# Parameters con cambio en k
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_1$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=18, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 18.4, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 18, "$E_0$", fontsize = 12, color = 'black')

# Texto agregado
plt.text(42, 12, '∆k', fontsize=12, color='black')
plt.text(45, 11.9, '↓', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())


ax1.legend()

plt.show()


# #### 2.3) ∆ Ms < 0; cantidad de dinero se disminuye: 

# $ Explicación: $
# 
# La disminución de la cantidad de dinero origina una disminución de la oferta real de dinero. Por ende, la oferta real de dinero será menor que la demanda real de dinero. El exceso de demanda que se produce en el mercado debe dar lugar a un aumento de la tasa de interés para que aumente la oferta y se restablezca el equilibrio en el mercado. En consecuencia, al disminuir la cantidad de dinero disminuye de 𝑀0𝑠 a 𝑀1𝑠, la recta de la oferta real de dinero se desplaza hacia la izquierda.
# 
# $ Graficando: $

# In[10]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P_1 = 20 
Y = 35
MS_0 = 550

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P_1, r, Y)
# Necesitamos crear la oferta de dinero.
MS_1 = MS_0 / P_1
MS


# In[11]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_1, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=18.1, xmin= 0, xmax= 0.50, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 19, "$r_0$", fontsize = 12, color = 'black')
ax1.text(52, 8, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(51, 18.5, "$E_0$", fontsize = 12, color = 'black')


# Nuevas curvas a partir del cambio en el nivel del producto
#ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS_1,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=22.5, xmin= 0, xmax= 0.30, linestyle = ":", color = "black")
ax1.text(0, 23, "$r_1$", fontsize = 12, color = 'black')
ax1.text(29, 8, "$(Ms/P)_1$", fontsize = 12, color = 'black')
ax1.text(28.5, 22.5, "$E_1$", fontsize = 12, color = 'black')

plt.text(32, 19.5, '∆Ms', fontsize=12, color='black')
plt.text(32.4, 18.8, '←', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## 3. Curva LM: 

# #### 3.1)

# In[38]:


#1: 

#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD

Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)

#2----------------------Curva LM

    # Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[39]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#CD5C5C')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=85, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 89, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 89, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 

#---------------------------------
    # Gráfico 2: Curva LM
    
ax2.set(title="LM SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax2.plot(Y, i, label="LM", color = '#3D59AB')

ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")

ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 122, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')

ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 122, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')

ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# ####  3.1.2)

# Analitico: 
# 
# $$M^s=M^d$$
# 
# A continuación se muestra el equilibrio entre la oferta real de dinero y la demanda monetaria. 
# 
# $$\frac{M^s_0}{P_0}=(kY-ji)$$
# 
# Ahora descompusimos cada una de sus variables
# 
# $$\frac{M^s_0}{P_0}=(kY-jr)$$
# 
# A continuación cambiamos i por r, pues en nuestra ecuación valen lo mismo
# 
# $$\frac{kY}{j}-\frac{M^s_0}{P_0}{j}=r$$
# 
# A continuación hacemos operaciones aritmeticas y nos queda la curva lm en donde r es la variable endogena. 
# 
# $$r= \frac{-1}{j}\frac{M^s_0}{P_0}+ \frac{kY}{j} $$
# 
# 

# #### 3.2) ¿Cuál es el efecto de una disminución en la Masa Monetaria?
# 
# Explica usando la intuición y gráficos.

# $$ ↓M^s_0 → ↓\frac{M^s_0}{P_0} → M^s < M^d → r↑ $$

# La masa monetaria cae, con lo cual la oferta monetaria cae y ello genera un desequilibrio. Para solucionarlo debe subir la tasa de interés. 

# In[4]:


Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 20

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[15]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(41, 56, '∆$M^s$', fontsize=12, color='black')
plt.text(36, 66, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Descenso en la Masa Monetaria $(M^s)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# #### 3.3)¿Cuál es el efecto de un aumento en k ? Explica usando intuición y gráficos.

# In[17]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 4

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[24]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(47, 140, '∆$k$', fontsize=12, color='black')
plt.text(47, 120, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Aumento en k $(k)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# Intuición: 
#     
# $$ ↑ k → ↑kY → M^d > M^s → ↑ r $$

# Cuando sube k sube la demanda monetaria. Ello quiere decir que la curva lm se encuentra en desequilibrio, para lo cual es necesario que la tasa de interés suba.
