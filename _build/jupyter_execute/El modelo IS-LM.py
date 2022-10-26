#!/usr/bin/env python
# coding: utf-8

# # El modelo IS-LM

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# Sebastian Torres Tapia - 20201586

# - Parte ensayo:
# 
# Mendoza, W. (2015). Demanda y oferta agregada en presencia de políticas monetarias no convencionales

# ¿De qué manera son afectadas la demanda y la oferta agregadas ante la presencia de políticas monetarias no convencionales? A partir de esta pregunta, el artículo busca modelar una política monetaria no convencional, extendiendo el modelo estándar de oferta y demanda de economía cerrada, para realizar la incorporación de innovaciones para el análisis de situaciones de riesgo posteriores a la crisis económica de 2008-2009. Con ello en mente, el autor utiliza como caso de análisis la Reserva Federal Estadounidense (con la tasa de interés de referencia como principal instrumento de política monetaria). La reducción dramática de esta tasa producto a la crisis obligó a la autoridad monetaria estadounidense a recurrir a los dos instrumentos de política poco convencionales que se desea presentar y analizar: el anuncio sobre el futuro de la trayectoria de la tasa de interés de corto plazo y la intervención (compra) directa en el mercado de bonos a largo plazo. 
# 
# En este proceso de análisis, son destacables dos aspectos como fortalezas. En primer lugar, la continuidad histórica: el modelo propuesto combina elementos de modelos tradicionales con métodos modernos para realizar un análisis integral de la situación actual y de las políticas macroeconómicas. Específicamente, el artículo recoge la teoría general de Keynes -que involucra instrumentos no convencionales-, los postulados de Tobin en el contexto de la crisis de la Gran Depresión y el modelo IS-LM propuesto por Hicks. En segundo lugar, es importante destacar su metodología: el desarrollo de la propuesta teórica se profundiza con abundantes gráficos y ecuaciones. Asimismo, el autor estructura el trabajo  de manera jerárquica entre los modelos macroeconómicos, plazos temporales, etc. En suma, dedica gran parte de su tiempo en “micro” explicaciones de conceptos y corpus teórico importantes para seguir la lógica planteada. 
# 
# No obstante, el enfoque planteado posee ciertas debilidades también. En primer lugar, las conclusiones a las que el autor arriba son bastante generales. A pesar de la profundidad dedicada a explicaciones teóricas o lógicas macroeconómicas respecto al proceso de análisis, se obvia el marco histórico y justificación que motivaron en principio dicho trabajo y que explicaría la importancia de la investigación respecto al estado del arte de estudios en políticas monetarias macroeconómicas. En segundo lugar, no se presenta la ejecución práctica  respecto a lo planteado por el autor. Por el momento, estamos ante una propuesta de modelo teórico provisional construida desde una perspectiva puramente occidental y norteamericana respecto al funcionamiento de la economía. En este sentido, resulta necesario seguir explorando su aplicación para diferentes regiones o casos de estudio de manera que se pueda fortalecer su hipótesis. 
# 
# En esta línea, el principal aporte del trabajo radica en su capacidad para reafirmar la utilidad y vigencia de propuestas teóricas del pasado para gestionar problemas macroeconómicos actuales. Asimismo, al plantear distintos enfoques de estudio (división por plazos, tipos de expectativas, etc), se puede aproximar cada vez más a la elaboración de mejores soluciones para posibles crisis futuras (predictibilidad) a partir de las lecciones del pasado. Asimismo, para tratar la compra de valores a largo plazo por parte de la reserva, se agrega un mercado de bonos a largo plazo al modelo Hicks-Hansen, en el que solo hay un mercado de bonos a corto plazo. 
# 
# Finalmente, considero que un paso necesario sería interpretar los resultados empíricos obtenidos a unidades de análisis concretas. Como destacan Camacho y González (2020), para determinar la validez metodológica del modelo, y resolver posibles contradicciones encontradas entre la teoría y la realidad, resulta importante realizar un análisis de política comparada entre casos para contrastar diferencias: países latinoamericanos o europeos, economías del G7, etc. Asimismo, dado que la economía, como ciencia social orientada al servicio de grupos socioeconómicos vulnerables, resulta de su interés colaborar con la búsqueda de soluciones que contribuyan a mitigar los efectos sociales y económicos del combate a la pandemia (De Rosa et al., 2020). En esta línea, el diseño de las políticas económicas de los diversos países frente a la crisis sanitaria del Covid-19 constituye un gran fenómeno de estudio para la aplicación del vasto contenido teórico planteado.
# 
# Bibliografía: 
# 
#     -Camacho, F., & González, J. (2020). Política monetaria y choques de oferta: El fin del superciclo de commodities en América Latina. Revista Económica de Centroamérica y República Dominicana, 1(1).
# 
#     -De Rosa, M., Lanzilotta, B., Perazzo, I., & Vigorito, A. (2020). Las políticas económicas y sociales frente a la expansión de la pandemia de COVID-19: aportes para el debate. Aportes y análisis en tiempos de coronavirus.  
# 

# ## Solución parte código:
# 
#     - Sebastian Torres
#     - Joaquín Del Castillo
# 
# ### Parte 1:
# 
# #### 1. Encuentre las ecuaciones de Ingreso  y tasa de interes  de equilibrio.
# 
# Derivación del equilibrio del modelo IS-LM:
# 
# 
# 
# En primer lugar vemos la ecuación de la curva IS,
# $$ r = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# - Donde $ B_0 = C_o + I_o + G_o + X_o $ y $ B_1 = 1 - (b - m)(1 - t) $
# 
# 
# Y, por otro lado, la ecuación de la curva LM:
# 
# $$  r = -\frac{1}{j}\frac{Mo^s}{P_o} + \frac{k}{j}Y $$
# 
# Podemos igualar, sustituir o reducir ambas ecuaciones para encontrar el nivel de Ingresos equilibrio $(Y^e)$ y la tasa de interés de equilibrio $(r^e)$:
# 
# 
# 
# $$ -\frac{1}{j}\frac{Mo^s}{P_o} + \frac{k}{j}Y = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# - Ingreso de equilibrio:
# 
# $$ Y^e = \frac{j B_o}{k h + j B_1} + (\frac{h}{k h + j B_1})\frac{Ms_o}{P_o} $$
# 
# - Tasa de interés de equilibrio:
# 
# $$ r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})\frac{Ms_o}{P_o} $$
# 
# Estas dos ecuaciones representan el modelo IS-LM
# 

# #### 2. Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[2]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 180            
P  = 25               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[3]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "C1") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "C0")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=51.5,  ymin= 0, ymax= 0.52, linestyle = ":", color = "black")
# Grafica la linea vertical - Y
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "black")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Grafico simultaneo en los mercados de dinero y de bienes", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ### Parte 2: Estática comparativa.

# #### 1) Analice los efectos sobre las variables endógenas Y, r de una disminución del gasto fiscal. El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
# 
# $$ G↓ → DA↓ → DA<Y → Y↓ → DA=Y $$
# 
# $$ Y↓ → Md↓ → Md < Ms → r↓ → Md=Ms $$
# 
#     - Matemático: 

# In[4]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (j*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (k*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)    


# In[5]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del producto con respecto al diferencial del gasto autonomo = ", df_Y_eq_Go)  # este diferencial es positivo


# In[6]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo = ", df_r_eq_Go)  # este diferencial es positivo


# Entonces: 
# 
# $$\frac{ΔY_e}{ΔG_0}=\frac{j}{kh+jB_1}ΔG_0<0 → Y (-)$$
# 
# $$\frac{ΔY_e}{ΔG_0}= (+)(-) < 0 → (-)$$
# 
# $$\frac{ΔY_e}{ΔG_0}=\frac{k}{kh+jB_1}ΔG_0 → 0$$
# 
# $$\frac{ΔY_e}{ΔG_0}= (+)(-) < 0 → (-)$$

#     - Gráfico: 

# In[7]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
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


# In[8]:


#--------------------------------------------------
    # NUEVA curva IS: reducción Gasto de Gobienro (Go)
    
# Definir SOLO el parámetro cambiado
Go = 30

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[9]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "C1") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "C1", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "C0")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=44.5,  ymin= 0, ymax= 0.45, linestyle = ":", color = "grey")
plt.axhline(y=78, xmin= 0, xmax= 0.45, linestyle = ":", color = "grey")

plt.axvline(x=53,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")
plt.text(42,70, '$E_1$', fontsize = 14, color = 'black')

plt.text(50,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#3D59AB')
plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

#plt.text(69, 115, '→', fontsize=15, color='grey')
plt.text(65, 60, '↙', fontsize=18, color='grey')

# Título, ejes y leyenda
ax.set(title="Politica Fiscal contractiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 2) Analice los efectos sobre las variables endógenas Y, r de una disminución de la masa monetaria (ΔMs<0). El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
#     
# $$Ms↓ → M°↓ → M°<Md → r↑$$
# 
# $$r↑ → I↓ → DA < Y → Y↓$$
# 
#     - Matemático: 

# In[10]:


Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[11]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)  # este diferencial es negativo


# In[12]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)  # este diferencial es positivo


# Entonces: 
# 
# $${Δr}=\frac{k}{kh+jB_1}ΔM_0 → 0$$
# 
# $$Δr= (+) * (-) = (-) > 0$$
# 
# $${ΔY_e}=\frac{j}{kh+jB_1}ΔMo<0$$
# 
# $$YΔ= (+) * (-) = (-) < 0$$

#     - Gráfico: 

# In[13]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.6
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
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


# In[14]:


# Definir SOLO el parámetro cambiado
Ms = 50

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[15]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "C1") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "C0")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "C0", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=50,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=93.5, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=52.5,  ymin= 0, ymax= 0.51, linestyle = ":", color = "grey")
plt.axhline(y=98, xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")
plt.text(62,87, '$E_1$', fontsize = 14, color = 'black')

#plt.axhline(y=68, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(55,95, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,85, '$r_0$', fontsize = 12, color = 'black')
plt.text(55,20, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 12, color = 'black')
plt.text(-1,100, '$r_1$', fontsize = 12, color = 'black')
plt.text(46,20, '$Y_1$', fontsize = 12, color = 'black')

#plt.text(69, 11, '→', fontsize=15, color='grey')
plt.text(52, 98, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efecto de un descenso de la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 3) Analice los efectos sobre las variables endógenas Y, r de un incremento de la tasa de impuestos  (Δts>0). El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
#     
# $$t↑ → DA↓ → DA<Y → Y↓ → DA=Y$$
# 
# $$Y↓ → Md↓ → Md < Ms → r↓ → Md=Ms$$
# 
#     - Matemático: 

# In[16]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[17]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del Producto con respecto al diferencial de la tasa de impuestos = ", df_Y_eq_t)  # este diferencial es negativo


# In[18]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interes con respecto al diferencial de la tasa de impuestos = ", df_r_eq_t)  # este diferencial es positivo


# Entonces: 
# 
# $${ΔY_e}=\frac{j}{kh+ Δt} + \frac{h}{kh + Δt} Δt>0$$
# 
# $$YΔ= {-} + {-}  = (-) < 0$$
# 
# $${Δr}=\frac{k}{kh+ Δt} + \frac{Δt}{kh + Δt} Δt> 0$$
# 
# $$Δr= {-} + {-}  = (-) < 0$$
# 
#     - Gráfico:

# In[19]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.3
t = 0.1

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
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


# In[20]:


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
t = 0.9

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[21]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "C1") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "C1", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "C0")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=54,  ymin= 0, ymax= 0.54, linestyle = ":", color = "grey")
plt.axhline(y=98, xmin= 0, xmax= 0.54, linestyle = ":", color = "grey")

plt.axvline(x=52,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")
plt.text(50,85, '$E_1$', fontsize = 14, color = 'black')

plt.text(52,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(55,10, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,85, '$r_1$', fontsize = 12, color = 'black')
plt.text(48,10, '$Y_1$', fontsize = 12, color = 'black')

#plt.text(69, 115, '→', fontsize=15, color='grey')
plt.text(58, 88, '↙', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Incremento de la tasa impositiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ## Puntos extra!

# ### Parte 1:
# 1. Encuentre las ecuaciones de Ingreso  y tasa de interes  de equilibrio.

# De la identidad Ingreso-Gasto:
# 
# $$ Y=C+I+G $$
# $$ Y=C_0+bY^d+I_0-hr+G_0 $$
# 
# Entonces, la identidad queda así:
# 
# $$ Y = (C_0 + I_0 - h(r) + G_0) + (b(1-t))Y $$
# 
# De forma corta:
# 
# $$ DA = α_0 + α_1Y $$
# 
# Donde $ α_0 = (C_0 + I_0 + G_0 -h(r))$ es el intercepto y $ α_1 = (b(1 - t)) $ es la pendiente de la función

# Determinando el ingreso de equilibrio: 
# 
# $$[1-(b)(1-t)]Y = C_0+I_0+ G_0 - h(r)$$
# $$ Y = \frac{1}{[1-(b)(1-t)]}(C_0+I_0+ G_0 - h(r)) $$

# Entonces, el modelo ingreso Ingreso-Gasto $ S(Y)=I(r)$:
# 
# $$[1-(b)(1-t)]Y - (C_0 + G_0) = I_0 - h(r) $$
# 
# Expresando la tasa de interés en función del ingreso, la tasa de interés de equilibrio es:
# 
# $$ r = \frac{1}{h}(C_0+I_0+G_0) - \frac{1-(b)(1-t)}{h}Y $$
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 

# Donde $ B_0 = (C_0 + I_0 + G_0))$ y $ B_1 = (1-b(1 - t)) $ 
# 
# Entonces: 
# 
# $$ (1) ... Y = \frac{B_0 - hr}{B_1} $$
# 

# Por otro lado, la ecuación de la curva LM: 
# 
# $$  \frac{Mo^s}{P_o} = kY - j(r+𝜋^e) $$
# 
# Entonces, dado que:
# 
# $$ i = (r+𝜋^e) $$
# 
# La ecuación quedaría así: 
# 
# $$ \frac{Mo^s}{P_o} = kY - ji $$ 
# 
# 
# $$ i = -\frac{1}{j}\frac{Mo^s}{P_o} + \frac{k}{j}Y $$
# 
# Entonces: 
# 
# $$ (2) ... Y = \frac{Mo^s}{(P_o)k} + \frac{j(i)}{k} $$
# 
# 
# Para encontrar el nivel de Ingreso de equilibrio y la tasa de interés de equilibrio, podemos igualar las ecuaciones de Ingreso de equilibrio y la tasa de interés de equilibrio. 
# 
# $$ -\frac{1}{j} \frac{Mo^s}{P_o} + \frac{k}{j}Y = \frac{B_0}{h} - {B_1} - {B_1}{h}Y $$
# 
# Entonces, la ecuacuión del Y de equilibrio quedaría: 
# $$ Y^e = \frac{jB_0}{kh + jB_1} - (\frac{B_1}{kh+jB_1}) \frac{Mo^s}{(P_o)}  $$
# 
# Asimismo, la tasa de interés de equilibrio: 
# 
# $$ r^e = B_0 - \frac{Mo^s(B_1) + B_1(j)(i)(P_0)}{P_o(k)} $$
# 
# Entonces, estas dos ecuaciones representan el modelo IS-LM. Ahora grafiquemos. 

# In[22]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.4
m = 0.5
t = 0.8
i = 0.2

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, h, Y, i):
    r_IS = (Co + Io + Go)/h - ( ( 1-b*(1-t)) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, h, Y, i)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 300             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[23]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "C1") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "C0")  #LM



# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=54.5,  ymin= 0, ymax= 0.54, linestyle = ":", color = "black")
# Grafica la linea vertical - Y
plt.axhline(y=93, xmin= 0, xmax= 0.54, linestyle = ":", color = "black")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="IS-LM Model", xlabel= r'Y', ylabel= r'r')
ax.legend()

ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

plt.show()


# ### Parte 2: Estática comparativa
# 
# #### 1. Analice los efectos sobre las variables endógenas Y, r de una disminución de los Precios $ (ΔP_0<0)$  . El análisis debe ser intuitivo, matemático y gráfico.
# 
#     - Intuitivo: 
#     
# $$P°↓ → Ms↑ → Ms>Md → r↓$$
# 
# $$r↓ → I↑ → DA > Y → Y↑$$
# 
#     - Matemático: 

#     - Gráfico:

# In[24]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.4
m = 0.5
t = 0.8
i = 0.2

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, h, Y, i):
    r_IS = (Co + Io + Go)/h - ( ( 1-b*(1-t)) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, h, Y, i)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 300             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r = r_LM( k, j, Ms, P, Y)


# In[25]:


#--------------------------------------------------
    # NUEVA curva LM: 
    
# Definir SOLO el parámetro cambiado
P = 5

# Generar la ecuación con el nuevo parámetro
def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_G = r_LM( k, j, Ms, P, Y)


# In[26]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "LM_(G_0)", color = "C0") #LM_orginal
ax.plot(Y, r_G, label = "LM_(G_1)", color = "C0", linestyle = 'dashed') #LM_modificada

ax.plot(Y, r_is, label="IS", color = "C1")  #IS_original

# Texto y figuras agregadas
plt.axvline(x=54.5,  ymin= 0, ymax= 0.62, linestyle = ":", color = "grey")
plt.axhline(y=93, xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")

plt.axvline(x=69,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=76, xmin= 0, xmax= 0.68, linestyle = ":", color = "grey")
plt.text(68,90, '$E_1$', fontsize = 14, color = 'black')

plt.text(50,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(48,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
plt.text(-1,70, '$r_1$', fontsize = 12, color = '#3D59AB')
plt.text(72,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

plt.text(60, 95, '→', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title=" Disminución de los precios", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ### 2. Analice los efectos sobre las variables endógenas Y, r de una disminución de la expansión esperada (Δπ<0). El análisis debe ser intuitivo, matemático y gráfico. 

# - Intuitivamente: 
# $$ π↓ → i↓ → Md > Ms → r↑ $$
# 
# $$ r↑ → I↓ → DA<Y → Y↓ $$

# - Matemáticamente: 
# $$ Δπ > 0 $$

# - Gráficamente: 

# In[27]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1 
Ms = 700   
P  = 20             

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

# Definir SOLO el parámetro cambiado
Ms = 100

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)

# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "C1") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "C0")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "C0", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=50.5,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.50, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=59.5,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=85, xmin= 0, xmax= 0.6, linestyle = ":", color = "grey")
plt.text(62,87, '$E_0$', fontsize = 14, color = 'black')

#plt.axhline(y=68, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(49,100, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#3D59AB')
#plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

plt.text(69, 115, '↑', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efectos de la disminucion de la inflacion esperada", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()

