#!/usr/bin/env python
# coding: utf-8

# # El modelo DA-OA

# Sebastian Torres - 20201586

# - Parte ensayo:
# 
# Dancourt, Oscar. (2014). Inflation Targeting in Peru: The Reasons for the Success

# ¿De qué manera se emplearon los principales instrumentos de la política monetaria peruana durante el periodo 2002-2013? A partir de esta pregunta, Dancourt se enfoca en la explicación de los canales a través de los cuales la política monetaria puede influir en la actividad económica y el nivel de precios. Identifica como los más importantes el canal del crédito y el canal del tipo de cambio. En este sentido, a partir del análisis del sistema económico dominado por bancos, el autor discute el uso e impacto de los principales instrumentos de política monetaria nacional: el coeficiente de reserva obligatoria en monedas locales y extranjeras, la intervención esterilizada en el mercado de divisas y la tasa de interés a corto plazo o la tasa de interés de referencia. Asimismo, se revisa el proceso de desdolarización del crédito bancario. En suma, se pretende explicar cómo el  Banco Central de Reserva (BCR) ha podido preservar la estabilidad macroeconómica tanto en contextos favorables como desfavorables.
# 
# En este artículo son destacables dos aspectos: la aplicación histórica y el recurso de citas y leyes de respaldo. En primer lugar, el autor es consciente de la necesidad de una explicación práctica de la carga teórica empleada. En este sentido, si bien se centra en una explicación del accionar del BCR, acompaña este proceso con el análisis de la realidad de la política monetaria durante un periodo concreto del pasado.  Particularmente, este proceso de revisión histórica resulta fundamental para la explicación del proceso de desdolarización del crédito (dividido en 3 etapas) y del canal del tipo de cambio (la lección de la crisis de 2008-2009). Asimismo, se recurre a la argumentación y contraargumentación del éxito de las herramientas de política monetaria empleadas a partir de la revisión de recesiones económicas o crisis bancarias internacionales y sus efectos en la esfera nacional. En todo caso, el contraste histórico aporta claridad al contenido análitico y teórico desarrollado. 
# 
# En segundo lugar, es destacable el empleo de citas de respaldo y el recurso de leyes o hipótesis para el sustento teórico del artículo en cuestión. A lo largo de su trabajo, Dancourt presenta y -más importante- discute con modelos teóricos sobre los cuales fundamenta su hipótesis: el modelo de Bernanke-Blinder y la regla de Taylor. Asimismo, presenta constantemente citas de respaldo (Ball, Yellen, Armas, Blanchard, etc) y de estudios internacionales realizados por instituciones de reputación como el Fondo Monetario Internacional o el propio BCR para apoyar su hipótesis. En suma, este constante intercambio de ideas aporta a la cohesión y comprensión del texto y de su validez académica.  
# 
# Sin embargo, existen debilidades que son necesarias mencionar. Por un lado, a nivel visual y de estructura, los gráficos empleados además de ser pocos, no son empleados efectivamente. El autor no desarrolla una interpretación matemática o lógica respecto a política monetaria para explicar las diversas acciones empleadas por el BCR y sus consecuencias en la economía nacional. Esto resulta aún más problemático cuando se identifica que el acceso a datos sobre recolección de divisas no suele estar disponible de manera pública para países emergentes. Por otro lado, como el mismo autor reconoce, resulta discutible generalizar el comportamiento del BCR dado que existen determinadas acciones que el enfoque no puede explicar: que los bancos centrales de diversos países que adquieren o acumulan divisas a ritmos diferenciados o que la compra de divisas no se encuentra necesariamente determinada por la magnitud de la reducción de la tasa de cambio, etc. 
# 
# A pesar de las debilidades encontradas, el artículo en cuestión constituye un aporte para la evaluación de políticas monetarias en respuesta a determinados tipos de crisis (particularmente frente a shocks externos). En esta línea, a partir del análisis y contraste temporal de los diversos periodos de crisis estudiados, se identifican medidas efectivas y compatibles que permiten combatir recesiones económicas internacionales. El grado de predictibilidad de las consecuencias y el enfoque de solución de situaciones son dos aportes importantes para la formación de tomadores de decisiones y asesores técnicos. 
# 
# Finalmente, considero que existen dos pasos para avanzar con esta investigación. En primer lugar, el mismo autor plantea la necesidad de comprobar si el esquema de política monetaria planteado -basado en un sistema de metas de inflación y acumulación de reservas de divisas- puede operar eficientemente en periodos de crisis prolongados o sin la tasa de interés de referencia como principal instrumento del BCR. En segundo lugar, como señalan Jaramillo y López (2021), resultaría un gran aporte actualizar este estudio al contexto inmediato de la pandemia por Covid-19 y la respuesta del BCR. Esto ya que fue durante este contexto que el BCR redujo su tasa de interés de referencia a su nivel mínimo histórico (0.25%) frente a la crisis económica y sanitaria. En este sentido, se debería profundizar en el estudio comparado sobre el manejo de la política monetaria que los diversos países latinoamericanos y sus bancos centrales ejecutaron frente a dicho contexto (Palomino, Mori, Rocca  et al., 2021). Dado que a nivel regional se desarrollaron medidas de integración como la creación de fondos y de apoyo de los diversos organismos internacionales, no se debe descuidar esta variable externa para la formulación de posibles soluciones. 
#  
# 
# Bibliografía: 
# 
# - Palomino Mendoza, M. del R., Mori Paredes, M. A., Rocca Carvajal, Y., Panche Rodriguez, O. B., & León Velarde, C. G. (2021). “Análisis de estrategias de integración económica en la región de América Latina y el Caribe para afrontar la pandemia de COVID-19, 2020”. Ciencia Latina Revista Científica Multidisciplinar, 5(3), 3735-3763. 
# 
#     https://doi.org/10.37811/cl_rcm.v5i3.561
# 
# - Jaramillo, M., & López, K. (2021). Políticas para combatir la pandemia de COVID-19. Lima: GRADE. 
# 
#     https://hdl.handle.net/20.500.12820/640
# 

# ### Parte código:
# 
# Integrantes: 
#     - Sebastian Torres 
#     - Joaquín Del Castillo 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd


# ## 1. Demanda Agregada $(DA)$:

# Matemáticamente, la forma de la función de demanda agregada se obtiene de las ecuaciones IS y LM, eliminando “r” y despejando P. Para efectuar esta operación se supondrá que P no es dado.

# A continuación la Curva IS:
# 
# $$ r = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_o + I_o + G_o + X_o $ y $ B_1 = 1 - (b - m)(1 - t)$

# Y, por otro, la ecuación LM
# 
# $$  r = -\frac{1}{j}\frac{Mo^s}{P} + \frac{k}{j}Y $$

# Eliminando “r” y despejando P, se obtiene:
# 
# $$  P = -\frac{h Mo^s}{-j B_o + (jB_1 + hk)Y} $$
# 
# O, en función del nivel de ingresos $(Y)$:
# 
# $$  Y = \frac{jB_o}{jB_1 + hk} + (\frac{hMo^s}{jB_1 + hk})\frac{1}{P} $$

# Ahora bien, teniendo en cuenta la ecuación de equilibrio en el mercado monetario
# 
# $$ Mo^s - P = kY - jr $$
# 
# Se reemplaza $(r)$, y se obtiene la ecuación de la demanda agregada $(DA)$
# 
# $$  P = \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y $$
# 

# ### 2. Oferta Agregada $(OA)$:

# #### - Oferta Agregada en el corto plazo:
# 
# El corto plazo es un periodo en el cual el producto $(Y)$ se ubica por debajo o por encima de su nivel de largo plazo o Producto Potencial $(\bar{Y})$.
# 
# Entonces, la curva de $OA$ de corto plazo se puede representar con la siguiente ecuación:
# 
# $$ P = P^e + θ(Y - \bar{Y}) $$ 
# 
# - Donde $(P)$ is the nivel de precios, $(P^e)$ el precio esperado y $\bar{Y}$ el producto potencial.

# ### 3.1 Ecuaciones de equilibrio DA-OA:
# ### El modelo DA-OA tiene tres variables endógenas. $$Y^{eq}, r^{eq} , P^{eq}$$

# La ecuación de la demanda agregada $(DA)$:
# 
# $$  P = \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y \ ...(1)$$ 
# 
# Y la ecuación de la oferta agregada $(OA)$:
# 
# $$ P = P^e + θ(Y - \bar{Y}) \ ...(2)$$ 

# - Para hallar $Y^{eq\_da\_oa}$ igualamos ecuaciones DA y OA:
# 
# $$ \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y = P^e + θ(Y - \bar{Y}) $$
# 
# $$ Y^{eq\_da\_oa} = [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ]*[(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})]$$

# - Para encontrar $P^{eq}$ reemplazamos $Y^{eq}$ en la ecuación de oferta agregada
# 
# $$ P^{eq\_da\_oa} = P^e + θ( Y^{eq\_da\_oa} - \bar{Y} ) $$ 
# 
# $$ P^{eq\_da\_oa} = P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ]*[(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} ) $$ 
# 
# 
# <!-- - Para hallar P^e, despejamos la ecuación de $OA$ en función de Y:
# 
# $$ P = P^e + θ(Y - \bar{Y}) $$ 
# 
# $$ Y = \frac{P - P^e - θ\bar{Y}}{θ} $$
# 
# Y reemplazamos $Y$ en la ecuación de $DA$:
# 
# $$ P^e = (\frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}) * (\frac{P - P^e - θ\bar{Y}}{θ}) $$ -->

# - Para encontrar $r^{eq\_da\_oa}$ solamente reemplazamos $P^{eq\_da\_oa}$ en la ecuación de tasa de interés de equilibrio del modelo IS-LM. 
# 
# - Tasa de interés de equilibrio:
# 
# $$ r^{eq\_is\_lm} = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})*(Ms_o - P)$$
# 
# - Tasa de interés de equilibrio en DA-OA
# $$ r^{eq\_is\_lm} = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})*(Ms_o - P^{eq\_da\_oa})$$
# 
# $$ r^{eq\_da\_oa} = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})*(Ms_o - P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ]*[(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} ) ) $$

# - Los valores de equilibrio de las tres principales variables endógenas 
# 
# 1. $$ Y^{eq\_da\_oa} = [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ]*[(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})]$$
# 
# 
# 2. $$ r^{eq\_da\_oa} = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})*(Ms_o - P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} 
# ]*\\
# [(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} ) ) $$
# 
# 
# 3. $$ P^{eq\_da\_oa} = P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ]*[(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} ) $$ 
# 

# - Equilibrio:

# In[2]:


#1--------------------------
    # Demanda Agregada
    
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

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada
    
# Parámetros

Y_size = 100

Pe = 100 
θ = 3
_Y = 20   

Y = np.arange(Y_size)


# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[3]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [P_AD[0], Y[0]] # DA, coordenada inicio
B = [P_AD[-1], Y[-1]] # DA, coordenada fin

C = [P_AS[0], Y[0]] # L_45, coordenada inicio
D = [P_AS[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[4]:


# Gráfico del modelo DA-OA

# Dimensiones del gráfico
y_max = np.max(P)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar 
ax.plot(Y, P_AD, label = "DA", color = "C4") #DA
ax.plot(Y, P_AS, label = "OA", color = "C8") #OA

# Líneas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1],  ymin= 0, ymax= 0.49, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 200, '$P_0$', fontsize = 12, color = 'black')
plt.text(53, 25, '$Y_0$', fontsize = 12, color = 'black')
plt.text(50, 202, '$E_0$', fontsize = 12, color = 'black')


# Eliminar valores de ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title="DA-OA", xlabel= r'Y', ylabel= r'P')
ax.legend()

plt.show()


# ## Parte 2: Estatica comparativa: 
# 
# ### 1. Analice los efectos sobre las variables endógenas Y, P y r de una disminución del gasto fiscal. $(ΔG_0 < 0) $.
# 
# ##### Intuitivo: 
# 
# Modelo IS-LM: 
# 
# $$G↓ → DA↓  → DA < Y → Y↓ $$
# 
# $$Y↓ → M_d↓  → M_d < Ms → r↓ $$
# 
# Modelo OA- DA:
# 
# $$Y↓ → Pe↓ → P↓$$
# 
# 
# ##### Matemáticamente: 
# 
# ##### IS-LM
# 
# 
# $$ {r_eΔ}=\frac{k}{kh+jB_1}ΔG_0 $$
# 
# $$r_eΔ= (+) * (-) = (-) < 0$$
# 
# $${Y_eΔ}=\frac{j}{kh+jB_1}ΔG_0<0$$
# 
# $$Y_eΔ= (+) * (-) = (-) < 0$$
# 
# 
# 
# ##### DA-OA:
# 
# $$ PΔ = P^e + θ(YΔ - \bar{Y}) $$ 
# 
# $$ PΔ = (+) + (+) * (-) = (-) < 0 $$
# 
# 
# ##### Gráfico: 

# In[5]:


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
Ms = 200             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva IS: disminucion en el gasto público (Go)

# Definir SOLO el parámetro cambiado
Go = 40

# Generar la ecuación con el nuevo parámetro
def r_IS_Go(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_Go = r_IS_Go(b, m, t, Co, Io, Go, Xo, h, Y)


# In[6]:


#1--------------------------
    # Demanda Agregada ORGINAL
    
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

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  


Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


#--------------------------------------------------
    # NUEVA Oferta Agregada

# Definir SOLO el parámetro cambiado

Go = 32

# Generar la ecuación con el nuevo parámetro

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD_Go(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Go =P_AD_Go(h, Ms, j, B0, B1, k, Y)


# In[7]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, i, label = "LM", color = "C1") #LM
ax1.plot(Y, r, label="IS", color = "C0")  #IS
ax1.plot(Y, r_Go, label="IS_Go", color = "C0", linestyle ='dashed')  #IS_Go


ax1.axvline(x=52.5,  ymin= 0, ymax= 0.53, linestyle = ":", color = "grey")
ax1.axvline(x=58,  ymin= 0, ymax= 0.58, linestyle = ":", color = "grey")
ax1.axhline(y=80,  xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")
ax1.axhline(y=90,  xmin= 0, xmax= 0.58, linestyle = ":", color = "grey")

ax1.text(50, 85, '↙', fontsize=15, color='grey')
ax1.text(54, -20, '←', fontsize=15, color='grey')
ax1.text(0, 82, '↓', fontsize=15, color='grey')
ax1.text(58, -25, '$Y_0$', fontsize=12, color='black')
ax1.text(45, -25, '$Y_1$', fontsize=12, color='C0')
ax1.text(0, 95, '$r_0$', fontsize=12, color='black')
ax1.text(0, 75, '$r_1$', fontsize=12, color='C0')


ax1.set(title="Efectos de una disminución en el Gasto Fiscal", xlabel= r'Y', ylabel= r'r')
ax1.legend()


#---------------------------------
    # Gráfico 2:

ax2.plot(Y, P_AS, label = "AS", color = "C4") #OA
ax2.plot(Y, P_AD, label = "AD", color = "C8") #DA
ax2.plot(Y, P_Go, label = "AD_Go", color = "C8", linestyle = 'dashed') #DA_Go

ax2.axvline(x=52,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=57,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=181,  xmin= 0, xmax= 0.58, linestyle = ":", color = "grey")
ax2.axhline(y=166,  xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

ax2.text(53, 10, '←', fontsize=15, color='grey')
ax2.text(55, 160, '↙', fontsize=15, color='grey')
ax2.text(0, 169, '↓', fontsize=15, color='grey')

ax2.text(58, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(45, 0, '$Y_1$', fontsize=12, color='C8')
ax2.text(0, 190, '$P_0$', fontsize=12, color='black')
ax2.text(0, 150, '$P_1$', fontsize=12, color='C8')

ax2.set(xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show


# --------

# 2. Analice los efectos sobre las variables endógenas Y, P y r de una disminución de la masa monetaria. $(ΔM^s < 0) $.
# 
# - Intuitivo: 

# 
# Modelo IS-LM: 
# 
# $$ M_s↓  → Ms < Md → r↑ $$
# 
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓ $$
# 
# Modelo OA- DA:
# 
# $$Y↓ → Pe↓ → P↓$$

# - Matemáticamente: 

# ##### IS-LM
# 
# 
# $${rΔ}=-\frac{k}{kh+jB_1}M_sΔ → 0$$
# 
# $$rΔ= (-)*(+) * (-) = (+) > 0$$
# 
# $${Y_eΔ}=\frac{j}{kh+jB_1}M_sΔ<0$$
# 
# $$YΔ= (+) * (-) = (-) < 0$$
# 
# 
# 
# ##### DA-OA:
# 
# $$ PΔ = P^e + θ(YΔ - \bar{Y}) $$ 
# 
# $$ PΔ = (+) + (+) * (-) = (-) < 0 $$
# 
# 
# 

# ##### Grafico: 

# In[8]:


# IS-LM

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
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM: incremento en la Masa Monetaria (Ms)

# Definir SOLO el parámetro cambiado
Ms = -100

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[9]:


#DA-OA

#1--------------------------
    # Demanda Agregada ORGINAL
    
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

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

Ms = 125

# Generar la ecuación con el nuevo parámetro

def P_AD_Ms(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Ms = P_AD_Ms(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[10]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "C1") #IS
ax1.plot(Y, i, label="LM", color = "C0")  #LM
ax1.plot(Y, i_Ms, label="LM_Ms", color = "C0", linestyle ='dashed')  #LM

ax1.axvline(x=45,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axvline(x=57,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axhline(y=88.9,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")
ax1.axhline(y=104,  xmin= 0, xmax= 0.46, linestyle = ":", color = "grey")

ax1.text(50, 110, '∆$M_s$', fontsize=12, color='black')
ax1.text(50, 102, '←', fontsize=15, color='grey')
ax1.text(50, 10, '←', fontsize=15, color='grey')
ax1.text(0, 94, '↑', fontsize=15, color='grey')
ax1.text(60, 15, '$Y_0$', fontsize=12, color='black')
ax1.text(40, 15, '$Y_1$', fontsize=12, color='C0')
ax1.text(0, 80, '$r_0$', fontsize=12, color='black')
ax1.text(0, 105, '$r_1$', fontsize=12, color='C0')


ax1.set(title="Efectos de un descenso en la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax1.legend()


#---------------------------------
    # Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "AD", color = "C4") #DA
ax2.plot(Y, P_Ms, label = "AD_Ms", color = "C4", linestyle = 'dashed') #DA_Ms
ax2.plot(Y, P_AS, label = "AS", color = "C8") #OA

ax2.axvline(x=44,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=56,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=140.5,  xmin= 0, xmax= 0.45, linestyle = ":", color = "grey")
ax2.axhline(y=178,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")

ax2.text(48, 0, '←', fontsize=15, color='grey')
ax2.text(36, 210, '∆$M_s$', fontsize=12, color='black')
ax2.text(36, 200, '←', fontsize=15, color='grey')
ax2.text(0, 155, '↓', fontsize=15, color='grey')

ax2.text(58, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(40, 0, '$Y_1$', fontsize=12, color='C4')
ax2.text(0, 190, '$P_0$', fontsize=12, color='black')
ax2.text(0, 125, '$P_1$', fontsize=12, color='C4')

ax2.set(xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show


# ### 3. Analice los efectos sobre las variables endógenas Y, P y r de un incremento en la tasa de impuestos. $(Δt > 0) $.
# 
# - Intuitivo: 

# 
# Modelo IS-LM: 
# 
# $$t↑ → Co↓ → DA↓ → DA < Y → Y↓ $$ #cambio más importante
# 
# $$Y↓ → M_d↓  → M_d < Ms → r↓ $$
# 
# Modelo OA- DA:
# 
# $$Y↓ → Pe↓ → P↓$$
# 

# - Matemáticamente :

# ##### IS-LM:
# 
# 
# $${ΔY_e}=\frac{j}{kh+ Δt} + \frac{h}{kh + Δt} Δt>0$$
# 
# $$ΔY= (-) + (-)  = (-) < 0$$
# 
# $${Δr}=\frac{k}{kh+ Δt} + \frac{Δt}{kh + Δt} Δt> 0$$
# 
# $$Δr= (-) + (-)  = (-) < 0$$

# ##### DA-OA:
# 
# $$ ΔP = P^e + θ(YΔ - \bar{Y})  Δt> 0 $$
# 
# $$ ΔP= (+) + (+) * (-) = (-) < 0$$

# In[11]:


# IS-LM

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
t = 0.5

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
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM: incremento en la tasa de impuestos

# Definir SOLO el parámetro cambiado
t = 0.9

# Generar la ecuación con el nuevo parámetro
def r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y)


# In[12]:


#DA-OA

#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.1

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

t = 0.99

# Generar la ecuación con el nuevo parámetro

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD_t(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_t = P_AD_t(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[13]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "C0") #IS
ax1.plot(Y, i, label="LM", color = "C1")  #LM
ax1.plot(Y, r_t, label="IS_t", color = "C0", linestyle ='dashed')  #LM

ax1.axvline(x=56.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axvline(x=58,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axhline(y=88.9,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")
ax1.axhline(y=91.5,  xmin= 0, xmax= 0.58, linestyle = ":", color = "grey")

ax1.text(55, 15, '←', fontsize=15, color='grey')
ax1.text(0, 85, '↓', fontsize=15, color='grey')
ax1.text(60, 15, '$Y_0$', fontsize=12, color='black')
ax1.text(50, 15, '$Y_1$', fontsize=12, color='C0')
ax1.text(0, 95, '$r_0$', fontsize=12, color='black')
ax1.text(0, 75, '$r_1$', fontsize=12, color='C0')


ax1.set(title="Efectos de un aumento en la tasa de tributación", xlabel= r'Y', ylabel= r'r')
ax1.legend()


#---------------------------------
    # Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "AD", color = "C4") #DA
ax2.plot(Y, P_t, label = "AD_t", color = "C4", linestyle = 'dashed') #DA_Ms
ax2.plot(Y, P_AS, label = "AS", color = "C8") #OA

ax2.axvline(x=57.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=56,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=182,  xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")
ax2.axhline(y=176.5,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")

ax2.text(55, 15, '←', fontsize=15, color='grey')
ax2.text(0, 174, '↓', fontsize=15, color='grey')

ax2.text(58, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(50, 0, '$Y_1$', fontsize=12, color='C4')
ax2.text(0, 190, '$P_0$', fontsize=12, color='black')
ax2.text(0, 158, '$P_1$', fontsize=12, color='C4')

ax2.set(xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show

