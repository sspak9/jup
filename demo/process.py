import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import csv

def get_rand_data(nrow, file_name):


    s = np.random.uniform( 0 , 1 , nrow * 10)
    for i in range(nrow * 10):
        s[i] = (int(s[i] * 10 + 0.5))/10

    rd = np.reshape( s , (nrow , 10))
    nd = np.zeros((nrow,13))


    for i in range(nrow):
        for j in range(10):
            nd[i][j] = rd[i][j]

    for i in range(nrow):
        
        race = rd[i][1]
        if race <= 0.33:
            nd[i][10] = 1
            nd[i][11] = 0
            nd[i][12] = 0

        elif race <= 0.66:
            nd[i][10] = 0
            nd[i][11] = 1
            nd[i][12] = 0

        else:
            nd[i][10] = 0
            nd[i][11] = 0
            nd[i][12] = 1


    '''
    0 - dosage: 80% max, above start taking 0.05 out of effectiveness
    1 - race: 0.33 , 0.66, 1 => 1,0,0 or 0,1,0, or 0,0,1
    2 - weight: over 80% -> negative 0.01.  if less than 0.5: dosage must be lower else negative 0.05
    3 - bp -> over 80% negative 0.01
    4 - age -> less than 0.5 -> negative 0.01. dosage if higher *0.1 * difference
    5,6 -> average two
    7 - 4*
    8 - * random gausian abs => * 0.01 of the new random
    9- same negative
    10,11,12 = race

    13 = dosage hit
    14 = weight hit
    '''

    ef = np.zeros((nrow,10))
    g = np.random.normal(0,0.1,nrow)
    r = np.zeros((nrow,))

    for i in range(nrow):
        #print('i=',i)

        # dosage
        dosage = nd[i][0]
        hit = 0
        if dosage >= 0.8:
            hit = (1-dosage) * -0.1
        else:
            hit = dosage / 10
        ef[i][0] = hit

        #race
        hit = nd[i][11]* 0.05 - nd[i][12]* 0.02
        ef[i][1] = hit

        #weight
        weight = nd[i][2]
        if weight < 0.2:
            weight = 0.2
            nd[i][2] = 0.2
        hit = 0
        if weight > 0.8:
            hit = (1-weight) * -0.05
        elif weight < 0.5:
            if dosage > 0.5:
                hit = - 0.1 * (dosage - weight)

        ef[i][2] = hit

        # bp
        hit = 0
        bp = nd[i][3]
        if bp < 0.25:
            bp = 0.25
            nd[i][3] = bp

        if bp > 0.8:
            hit = ( 1- bp) * -0.05

        ef[i][3] = hit

        # age
        age = nd[i][4]
        if age < .21:
            age = 0.21
            nd[i][4] = 0.21
        hit = 0
        if age < 0.5:
            if dosage > 0.5:
                hit = - 0.1 * (dosage - age)

        ef[i][4] = hit

        '''
        ef[i][5] = (nd[i][5] + nd[i][6]) / 2

        ef[i][7] = nd[i][7]*nd[i][7]*nd[i][7]*nd[i][7]

        ef[i][8] = nd[i][8] *g[i]
        #ef[i][9] = nd[i][9]* -1 * abs(g[i])
        '''

        for j in range(10):
            r[i] += ef[i][j]

    #print(r)
    #sns.set(color_codes=True)
    #sns.distplot(r, kde=False, rug=True)
    #sns.distplot(r)
    #plt.show()
    result = np.zeros((nrow,))
    for i in range(nrow):
        if r[i] > 0.075:
            result[i] = 2
        elif r[i] > -0.05:
            result[i] = 1

    #sns.distplot(result)
    #plt.show()
    # write csv
    # dosage , race1 , race2 , race3 , weight , bp , age
    with open(file_name , 'w') as w:
        w.write('dosage (max:100 units),race1,race2,race3,weight (max: 300 lbs),bp (max:300/100),age (max:80),effective,no effect,side effect\n')
        for i in range(nrow):
            
            line = (str(nd[i][0]) + ',' +
                    str(nd[i][10]) + ',' +
                    str(nd[i][11]) + ',' +
                    str(nd[i][12]) + ',' +
                    str(nd[i][2]) + ',' +
                    str(nd[i][3]) + ',' +
                    str(nd[i][4]) + ',' 
            )
            if result[i] == 2:
                line += '1,0,0\n'
            elif result[i] == 1:
                line += '0,1,0\n'
            else:
                line += '0,0,1\n'
            w.write(line)


if __name__ == '__main__':
    get_rand_data(nrow=500,file_name='validation.csv')
    get_rand_data(nrow=10000, file_name='data.csv')
