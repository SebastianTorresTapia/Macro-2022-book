#!/usr/bin/env python
# coding: utf-8

# # Reporte 2

# Sebastian Torres Tapia - 20201586

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel


# - Parte ensayo:
# 
# Martinelli, C., & Vega, M. (2018). The Monetary and Fiscal History of Peru, 1960-2017. Encuentro, (1/33).

# El artículo intenta explicar cómo la monetización de la deuda en un régimen de dominio fiscal se configuró como detonante de la inflación que vivió el Perú entre 1970 y 1980. Para ello, se realiza una revisión histórica de las políticas monetarias y fiscales ejecutadas durante los diversos gobiernos y sus efectos acumulados hasta llegar a la hiperinflación.
# 
# En esta línea, a lo largo del artículo se identifican ciertas fortalezas. En primer lugar, se realizan dos ejercicios de medición para explorar y fundamentar sus hipótesis.  Si bien existe un notable apego hacia las explicaciones teóricas (evidenciado en secciones enteras destinadas a explicar modelos matemáticos), es destacable el esfuerzo para describir las particularidades de la economía peruana. A través de un ejercicio de contabilidad del crecimiento (desglosando los cambios en el PIB por trabajador en varios componentes) y otro de contabilidad escalar (desglosando el financiamiento del gobierno en sus diversas fuentes), los autores explican sus herramientas para la comprensión del caso peruano y el periodo estudiado. 
# En segundo lugar, el artículo posee un enfoque histórico que permite analizar las políticas económicas implementadas y los factores externos a partir de los diversos períodos de gobierno. En este sentido, aportan datos importantes del escenario internacional para hacer un análisis completo de las causas de la inflación en el país. Además, se realizan comparaciones estadísticas e históricas con el período anterior y posterior al estudiado. Este esfuerzo por no aislar las explicaciones macroeconómicas teóricas y los procesos históricos causales es destacable. No obstante, debe ser tomado de manera crítica frente a posibles sesgos (especialmente frente a los periodos de reformas radicales y autoritarias). 
# 
# Sin embargo, resulta necesario identificar ciertas debilidades presentes. Si bien los autores brindan explicaciones de la metodología y las dos herramientas empleadas, el lenguaje empleado para la explicación de los modelos matemáticos resulta técnico. Ello se traduce en una mayor dificultad para seguir los mecanismos de análisis y argumentos. Asimismo, es posible evidenciar la ausencia de factores explicativos fuertemente relacionados a algunas variables empleadas. Por ejemplo, la recolección de datos sobre la volatilidad de las materias primas -frecuentemente empleada como factor de causalidad-, la ocurrencia de desastres naturales o de los efectos de las movilizaciones internas (especialmente durante los periodos de transiciones democráticas post autoritarismos) se encuentra ausente. La consideración de factores sociales e incluso demográficos como la reforma agraria o las movilización urbana durante el terrorismo tienen efectos en componentes como el gasto público, el nivel de inversión pública y privada, etc. 
# 
# Respecto a las contribuciones de este artículo y sus resultados, encontramos dos principales. En primer lugar, mediante el enfoque histórico brindado, se identifica cómo, en cierta medida, la afinidad política del gobierno condiciona a su vez el modelo económico aplicado en la economía. En segundo lugar, si bien el artículo posee un enfoque principalmente teórico, los autores establecen breves reflexiones sobre posibles soluciones adaptables para mantener un escenario más favorable para la economía frente a situaciones de crisis. En suma, ambos aportes resultan interesantes para la investigación sobre el impacto de la perspectiva ciudadana en la salud macroeconómica del país. 
# 
# Finalmente, considero que el desarrollo de esta pregunta puede avanzar a partir del análisis de fenómenos como el populismo y su relación con diversos espectros políticos y con la ciudadanía. Como destaca Edwards (2019), los populistas utilizan una retórica feroz para enfrentar los intereses del "pueblo" contra los de los bancos o las grandes empresas. Asimismo, en base al amplio apoyo de sus bases, estos líderes carismáticos suelen impulsar políticas redistributivas que violan las leyes básicas de la economía y, en particular, las restricciones presupuestarias para asegurar su respaldo. Para evitar este tipo de escenarios, resulta necesario el diseño y promoción de reformas para atender las brechas sociales, pero sin comprometer el marco macroeconómico que ha promovido exitosamente los avances de las últimas décadas (Burrows 2020; Ortiz y Winkelried 2021). La mejor manera de lograr ello es mediante una mayor inversión en educación para la población con el objetivo de incentivar una ciudadanía alerta y consciente respecto al crecimiento, desarrollo, inclusión y fiscalización política y económica. 
#  
# 
# - Bibliografía:
# 
# Burrows, D. (2020). Causes of and Barriers Within the Informal Economy: How to Promote Change for Women Domestic Workers in Lima, Peru.
# 
# Edwards, S. (2019). On Latin American populism, and its echoes around the world. Journal of Economic Perspectives, 33(4), 76-99.
# 
# Ortiz, M., & Winkelried, D. (2021). El largo camino hacia la estabilidad macroeconómica. En búsqueda de un desarrollo integral, 20, 27-54.
# 

# ### Parte código 

# ##### Integrantes: 
# - Sebastian Torres (20201586)
# - Joaquin Del Castillo (20201448)

# #### Pregunta 1

# ##### 1.A)

# La curva IS se deriva de la igualdad entre el ingreso  y la demanda agregada :
# 
# $$ DA = C + I + G + X - M $$
# 
# Donde:
# 
# $$ C = C_0 + bY^d $$
# 
# $$ I = I_0 - hr $$
# 
# $$ G = G_0 $$
# 
# $$ X = X_0 $$
# 
# $$ M = mY^d $$
# 
# $$ T = tY$$
# 
# 
# Para llegar al equilibrio Ahorro-Inversión, debemos restar la tributación  de ambos miembros de la igualdad.
# 
# 
# $$Y-T= C+I-T+G+X-M$$
# 
# $$Y^d= C+I-T+G+X-M$$
# 
# Esta igualdad se puede reescribir de la siguiente forma:
# 
# $$(Y^d-C) + (T-G) + (M-X)= I$$
# 
# Las tres partes de la derecha constituyen los tres componentes del ahorro total : ahorro privado , ahorro del gobierno  y ahorro externo :
# 
# $$S= Sp + Sg + Se$$
# 
# Entonces, el ahorro total es igual a la inversión
# 
# $$Sp + Sg + Se= I$$
# 
# $$S(Y)= I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$S_p + S_g + S_e = I_0 - hr$$
# 
# $$(Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 - hr$$
# 
# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso,
# 
# 
# $$[1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
#     
# $$ hr = (C_0 + G_0 + I_0 + X_0) - (1 - (b - m)(1 - t))Y $$
#     
# $$ r =\frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ es el intercepto y $  B_1 = 1 - (b - m)(1 - t) $ es la pendiente.
# 
# 

# ##### 1.B)

# Utilizando:
# 
# $$ Y =\frac{1}{1 - (b - m)(1 - t)}(C_0 + G_0 + I_0 + X_0) - \frac{h}{1 - (b - m)(1 - t)}r $$
# 
# o, lo que es lo mismo:
# 
# $$ r =\frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$

# In[3]:


Co, Io, Go, Xo, h, b, m, t, Y = symbols('Co Io Go Xo h b m t Y')

f = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h


df_t = diff(f, Y)
df_t #∆r/∆Y


# Considerando el diferencial de $ ∆r/∆Y $ (que representa la pendiente):
# 

# Sabiendo que:
# 
# - b > m, entonces $ (b-m) > 0 $
# 
# - Los componentes autónomos no cambian: $∆C_0 = ∆I_0 = ∆X_0 = ∆h = 0$
# 

# $Entonces: $ 
# 
# $$ \frac{∆r}{∆Y} = - (\frac{1-(b-m)(1-t)}{h}) $$
# 
# Analizando, 
# 
# $$ (\frac{1-(b-m)(1-t)}{h}) $ > 0 $$
# 
# Por lo tanto: 
# 
# $$ \frac{∆r}{∆Y} = (-)(+) $$
# 
# $$ \frac{∆r}{∆Y} = (-); < 0 $$

# ##### 1.C) 

# ###### Explicación de la derivación de la curva IS a partir del equilibrio $ Y=DA $

# Recordemos la ecuación del ingreso de equilibrio a corto plazo que fue obtenida a partir del equilibrio $ (Y=DA) $ 
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# Esta ecuación, después de algunas operaciones, puede expresarse en función de la tasa de interés $(r)$:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Simplificada:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ y $  B_1 = 1 - (b - m)(1 - t) $
# 
# Interpretación: Esta ecuación indica pares de valores ordenados de ingreso (𝑌) y de tasa de interés (𝑟) que equilibran el mercado de bienes. Esta relación inversa entre el producto y el nivel de ingresos se puede ilustrar gráficamente a partir del equilibro 𝐷𝐴=𝑌 en la recta de 45°.
# 
# 
# Entonces, Para la derivación gráfica, se tiene que recordar la ecuación de la Demanda Agregada (DA):

# - Demanda Agregada:

# In[4]:


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

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)

#--------------------------------------------------
# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)

#--------------------------------------------------
# Segunda curva de ingreso de equilibrio

    # Definir cualquier parámetro autónomo
Go = 35

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# - Curva IS

# In[5]:


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

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[6]:


# Gráfico de la derivación de la curva IS a partir de la igualdad (DA = Y)

    # Dos gráficos en un solo cuadro (ax1 para el primero y ax2 para el segundo)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: ingreso de Equilibrio
ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.plot(Y, DA_IS_K, label = "DA_0", color = "C0") 
ax1.plot(Y, DA_G, label = "DA_1", color = "C0") 
ax1.plot(Y, L_45, color = "#404040") 

ax1.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
ax1.axvline(x = 80,  ymin= 0, ymax = 0.79, linestyle = ":", color = "grey")

ax1.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
ax1.text(2.5, -3, '$◝$', fontsize = 30, color = 'black')
ax1.text(77, 0, '$Y_1$', fontsize = 12, color = 'black')
ax1.text(66, 0, '$Y_0$', fontsize = 12, color = 'black')
ax1.text(65, 150, 'E_0', fontsize = 12, color = 'black')
ax1.text(80, 154, 'DA(r0)', fontsize = 12, color = 'black')
ax1.text(82, 194, 'E_1', fontsize = 12, color = 'black')
ax1.text(90, 194, 'DA(r1; siendo r1 < r0)', fontsize = 12, color = 'black')

ax1.set(title = "Derivación de la curva IS a partir del equilibrio $Y=DA$", xlabel = r'Y', ylabel = r'DA')
ax1.legend()

#---------------------------------
    # Gráfico 2: Curva IS

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.plot(Y, r, label = "IS", color = "C1") 

ax2.axvline(x = 80, ymin= 0, ymax = 0.99, linestyle = ":", color = "lightgray")
ax2.axvline(x = 70,  ymin= 0, ymax = 0.99, linestyle = ":", color = "lightgray")
plt.axhline(y = 151.5, xmin= 0, xmax = 0.69, linestyle = ":", color = "lightgray")
plt.axhline(y = 143, xmin= 0, xmax = 0.79, linestyle = ":", color = "lightgray")

ax2.text(72, 128, '$Y_0$', fontsize = 12, color = 'black')
ax2.text(80, 128, '$Y_1$', fontsize = 12, color = 'black')
ax2.text(1, 153, '$r_0$', fontsize = 12, color = 'black')
ax2.text(1, 145, '$r_1$', fontsize = 12, color = 'black')
ax2.text(72, 152, 'E_0', fontsize = 12, color = 'black')
ax2.text(83, 144, 'E_1', fontsize = 12, color = 'black')

ax2.legend()

plt.show()


#  ##### Pregunta 2:

# La igualdad de ingreso- ahorro empieza restando T en ambos lados, ello debido a que de esa forma podemos hallar realmente el ingreso disponible: 
# 
# $$Y-T= C+I-T+G+X-M$$
# 
# $$Y^d= C+I-T+G+X-M$$
# 
# Esta igualdad se puede reescribir de la siguiente forma:
# 
# $$(Y^d-C) + (T-G) + (M-X)= I$$
# 
# Las tres partes a la derecha de la igualdad constituyen los tres componentes del ahorro total : ahorro privado , ahorro del gobierno  y ahorro externo :
# 
# $$S= Sp + Sg + Se$$
# 
# Entonces, el ahorro total es igual a la inversión
# 
# $$Sp + Sg + Se= I$$
# 
# $$S(Y)= I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$S_p + S_g + S_e = I_0 - hr$$
# 
# $$(Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 - hr$$
# 
# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso,
# 
# 
# $$[1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
#     
# $$ hr = (C_0 + G_0 + I_0 + X_0) - (1 - (b - m)(1 - t))Y $$
#     
# $$ r =\frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ y $  B_1 = 1 - (b - m)(1 - t) $

# Graficando la curva IS de equilibrio en el Mercado de Bienes:

# In[7]:


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

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)





# Gráfico de la curva IS

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "#8B0A50") #Demanda agregada

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title = "Curva IS de Equilibrio de Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ##### Pregunta 3: 

# In[8]:


z_size = 100          
z = np.arange(z_size) 
z 


# In[9]:


#Parameters
Y_size = 100 
b = 0.2                 # propensión marginal a consumir
m = 0.2                 # propensión a importar
t = 1                   # tasa impositiva
Co = 20                 # consumo autónomo
Io = 10                 # inversión autónoma
Go = 8                  # gasto gobierno 
Xo = 2                  # exportación 
h =  5                  # constante de decisión a invertir

Y = np.arange(Y_size)


# In[10]:


def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS


# In[11]:


def Y_IS(b, m, t, Co, Io, Go, Xo, h, r):
    Y_IS = (Co + Io + Go + Xo - h*r)/(1-(b-m)*(1-t))
    return Y_IS


# In[12]:


r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
r


# $ Graficando: $

# In[13]:


y_max = np.max(r)
x_max = Y_IS(b, m, t, Co, Io, Go, Xo, h, 0)


v = [0, x_max, 0, y_max]                        
fig, ax = plt.subplots(figsize=(10, 8))
ax.set(title="IS SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax.plot(Y, r, "k-")

plt.axvline(x = 8, ymin= 0, ymax = 0.60, linestyle = ":", color = "grey")
plt.axvline(x = 16,  ymin= 0, ymax = 0.60, linestyle = ":", color = "grey")
plt.axvline(x = 24, ymin= 0, ymax = 0.60, linestyle = ":", color = "grey")
plt.axhline(y = 4.8, xmin= 0, xmax= 0.60, linestyle = ":", color = "grey")


plt.text(-2, 4.7, '$r_A$', fontsize = 12, color = 'black')
plt.text(7.2, -0.3, '$Y_C$', fontsize = 15, color = 'black')
plt.text(15.5, -0.3, '$Y_A$', fontsize = 15, color = 'black')  
plt.text(23.3, -0.3, '$Y_B$', fontsize = 15, color = 'black') 
plt.text(7.2, 4.9, '$C$', fontsize = 15, color = 'black')
plt.text(15.5, 4.9, '$A$', fontsize = 15, color = 'black')  
plt.text(23.7, 4.9, '$B$', fontsize = 15, color = 'black') 
plt.text(25, 5.5, 'Exceso  de  oferta', fontsize = 15, color = 'black')  
plt.text(7, 2.5, 'Exceso de demanda', fontsize = 15, color = 'black') 

ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())
    
plt.axis(v)                                    
plt.show()



# $ Entonces: $ 

# Como se puede observar, todos los puntos de la curva IS corresponden a pares ordenados que equilibran el mercado de bienes, donde el punto "A" representa un par ordenado que equilibra el ahorro con la inversión (I=S). Por ende, los puntos fuera de dicha curva ("C" y "B") son de desequilibrio en el mercado de bienes. 
# 
# Por un lado, en el punto "C" $ (Y_C, r_A) $ la inversión se mantiene constante (dado que se mantiene la tasa de interés), pero el ahorro es menor respecto al ahorro del punto "A" dado que el ingreso es menor también. Entonces, en el punto "C", $I_A > S_C $; por tanto, a la izquierda de la curva de IS el desequilibrio del mercado de bienes es de exceso de demanda. Por otro lado, en el punto "B" $ (Y_B, r_A) $ la inversión sigue siendo constante pero el ingreso y, por ende, el ahorro son mayores que respecto a "A". Así, en "b" $I_A < S_B $; por tanto, a la derecha de la curva de IS el desequilibrio del mercado es de exceso de oferta. 

# #### Pregunta 4:

# ##### 4.A)

# Creando la curva IS

# In[14]:


# Curva IS ORIGINAL

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

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
  r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
  return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
  # NUEVA curva IS
  
# Definir SOLO el parámetro cambiado
Go=40

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
  r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
  return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# Análisis gráfico

# In[15]:


# Graph

# Dimensions
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curves to plot
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_G, label = "IS_G", color = "C1", linestyle = 'dashed') #New IS

# Text added
plt.text(47, 162, '∆Go', fontsize=12, color='green')
plt.text(49, 159, '←', fontsize=15, color='grey')

# Title and legend
ax.set(title = "Baja gasto de gobierno $(G_0)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# Análisis intuitivo:

# $$ G_0↓ → G↓ → DA↓ → DA < Y → Y↓ $$
# 
# $ Entonces: $ Si cae el Gasto de Gobierno producto de una Política Fiscal Contractiva, la Demanda Agregada caerá dado que las personas tendrán menos dinero disponible. Por tanto, la nueva Demanda Agregada será menor que el nivel de producción. Entonces, se deberá disminuir el nivel de producción para volver al equilibrio. 
# 

# ##### 4.B)

# Análisis gráfico:

# In[16]:


# ORIGINAL IS Curve

# Parameters

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Equation 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # New IS Curve

# Define ONLY the changed parameter
t = 0.2

# Equation with the changed parameter
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)

# Graph

# Dimensions
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curves to plot
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_G, label = "IS_t", color = "C1", linestyle = 'dashed') #New IS

# Text added
plt.text(47, 130, '∆t', fontsize=12, color='black')
plt.text(50, 140, '→', fontsize=15, color='grey')

# Title and legend
ax.set(title = "Baja la tasa de impuestos $(t)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# $$ t↓ → C↑ → DA↑ → DA > Y → Y↑ $$
# 
# $ Entonces: $ Si cae la tasa de impuestos (t), la cantidad de dinero que la gente tendrá será mayor (dado que el Estado les quitará menos en impuestos). En este sentido, la Demanda Agregada incrementará, siendo mayor al nivel de producción inicial. En este sentido, el nivel de producción deberá subir para volver al equilibrio. Se da una expansión del mercado de bienes a partir de la pendiente. 

# ##### 4.C)

# Análisis gráfico:

# In[17]:


# ORIGINAL IS Curve

# Parameters

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Equation 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
  r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
  return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
  # New IS Curve

# Define ONLY the changed parameter
b = 0.6

# Equation with the changed parameter
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
  r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
  return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)







 # Graph

# Dimensions
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curves to plot
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_G, label = "IS_B", color = "C1", linestyle = 'dashed') #New IS

# Text added
plt.text(42, 130, '∆b#####', fontsize=12, color='black')
plt.text(42, 140, '←', fontsize=15, color='grey')

# Title and legend
ax.set(title = "Baja la propensión marginal a consumir $(b)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()



#  Análsis intuitivo:  
# 
# $$ ↓b → ↓C → DA↓ → DA < Y → ↓Y $$
# 
# $ Entonces: $ Si cae la propension marginal a consumir, se traduce en una caída el consumo. En consecuencia, si cae el consumo, la Demanda Agregada cae. Por ende, la Demanda Agregada será menor que el nivel de producción inicial, razón por a cual deberá reducirse el nivel la producción para volver al equilibrio. 
