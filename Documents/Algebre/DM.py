# -*- coding: utf-8 -*-


import csv
import numpy as np 
import matplotlib as plt
import pylab as pl
import math

with open("data.tab", 'rb') as f:
    reader = csv.DictReader(f, delimiter = '\t')
'''    
    for row in list(reader):
        for key, value in row.iteritems():
            if key !='cell' and value[0] not in ['H','N','P']:
                print(key,int(float(value)))
            
            

            # create a dict out of reader, converting all values to
            # integers
     baseUserItem =  [dict([key, int(value)] for key, value in row.iteritems()) for row in list(reader)]
'''


def noyau(x,y,e):
    return math.exp((-1.0/e)*(math.hypot(x,y)**2))




def valeurs(x,e): #  Prend en argument un jeu de données x et un paramètre e


    L=np.zeros((len(x),len(x)))
    for i in range (0,len(x)):
        for j in range(0,len(x)):
            L[i][j]=noyau(x[i],x[j],e) #Calcul de L avec la fonction noyau
    D=np.zeros((len(x),len(x)))
    for i in range(0,len(x)):
        D[i][i]=sum([L[i][j] for j in range (0,len(x))]) #Calcul de D a partir de L


    Dp=np.zeros((len(x),len(x)))# Calcul des puissances de D ( en évitant la division par 0)
    Dm=np.zeros((len(x),len(x)))

    for i in range (0,len(x)):
        Dp[i][i]=D[i][i]**(-0.5)
        Dm[i][i]=D[i][i]**(0.5)
    
    Ms=np.dot(np.dot(Dp,L),Dm) # Calcul de la matrice Ms

    vlp,vect=np.linalg.eig(Ms)
    vlp=sorted(abs(vlp),reverse=True) # récupération des valeurs propres
    #comme une liste de valeurs absolues dans l'ordre décroissant.
    

    M=np.dot(np.linalg.inv(D),L) # Calcul de M
    

    phi=[] # On remplit la liste phi des vecteurs propres
    for j in range(len(x)):
        phi.append([i*math.sqrt(D[j][j]) for i in vect[j]])
    

    return vlp,phi # La fonction retourne les valeurs et vecteurs
# propres calculés

def diffusion_map(vlp, phi, k, t): # Calcule la diffusion map à partir
#des valeurs et vecteurs propres calculés précédemment et de deux paramètres

    
    psi=[] # liste contenant les valeurs de la diffusion map
    for j in range(0,k-1):
        psi.append([i*(vlp[j]**t) for i in phi[j]])# Calcul des valeurs 


    
    psi=np.reshape(np.asarray(psi),(len(phi), len(psi)))# On change psi en tableau
    
    return psi # retourne la diffusion map
        



vlp,phi=valeurs([5,8,9,5,6,9,20],4)
print(diffusion_map(vlp, phi, 5, 2))





