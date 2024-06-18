import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from readCsv import readResult, readScore

#if I get this to work, this will graph genes vs scores, with the ability to put multiple on top of each other. That last part doesn't work, it just makes a second graph next to the first at the moment
def graphScores(file1, file2 = '', file3 = ''): 
    #Input: csv files that are archetype/gene score tables (1-3), if multiple, they need to be about the same genes and archetypes
    #Output: a graph of the data. as described above the definition line
    g1, a1, d1 = readResult(file1)
    x = g1
    dt1=np.transpose(d1)
    #if you want to graph data from a different archetype, change the 0 to the number that matches that archetype number
    #I can't graph them all because the csv file I'm using has almost 600 genes, so graphing 600 points alone is a lot, I don't want
    #to also multiply that by 11 archetypes
    arche = 6
    y = dt1[arche]
    #plt.plot(x, y)
    fig = plt.figure(figsize = (25, 5))
    ax = fig.add_subplot()
    ax.plot(x, y)
    ax.tick_params(axis='x', rotation=70)
    ax.set_xticks([])
    ax.set_title('Graph of data using archetype ' + a1[arche])
    if file2:
        g2, a2, d2 = readResult(file2)
        x = g2
        dt2=np.transpose(d2)
        y = dt2[arche]
        ax.plot(x,y)

    plt.show()
    return

#This graph compares heuristic scores to exhaustive scores, a perfect heuristic would be a straight "45 degree" line
def graphExhaustHeurist(exhaustFile, heuristFile):
    #Input: the exhaustive score csv file and the heuristic score csv file. These are the files generated by my GeneArchetype_Heuristic_Table or the exhaustive one, or a table of similar format
    #output: a graph of the data, as described above the definition line
    eGene, eArch, eData = readResult(exhaustFile)
    hGene, hArch, hData = readResult(heuristFile)

    exhaust = eData.flatten()
    heurist = hData.flatten()

    plt.scatter(exhaust, heurist, s=.5)
    plt.show()
    return

#this chart will have the mean and standard dev of 
def correctChart(exhaustFile, heuristFile):
    eGene, eArch, eData = readResult(exhaustFile)
    hGene, hArch, hData = readResult(heuristFile)

    exhaust = eData.flatten()
    heurist = hData.flatten()

    a = exhaust<.05
    e1 = exhaust[a]    
    m1 = np.mean(e1)
    s1 = np.std(e1)

    a1 = exhaust > .05
    a2 = exhaust < .1
    a = a1*a2
    e2 = exhaust[a]
    m2 = np.mean(e2)
    s2 = np.std(e2)

    a1 = exhaust > .1
    a2 = exhaust < .15
    a = a1*a2
    e3 = exhaust[a]
    m3 = np.mean(e3)
    s3 = np.std(e3)

    a1 = exhaust > .15
    a2 = exhaust < .2
    a = a1*a2
    e4 = exhaust[a]
    m4 = np.mean(e4)
    s4 = np.std(e4)

    a1 = exhaust > .4
    a2 = exhaust < .45
    a = a1*a2
    e5 = exhaust[a]
    m5 = np.mean(e5)
    s5 = np.std(e5)

    a1 = exhaust > .45
    a2 = exhaust < .5
    a = a1*a2
    e6 = exhaust[a]
    m6 = np.mean(e6)
    s6 = np.std(e6)
    
    a1 = exhaust > .5
    a2 = exhaust < .55
    a = a1*a2
    e7 = exhaust[a]
    m7 = np.mean(e7)
    s7 = np.std(e7)

    values = [0, .05, .1, .15, .4, .45, .5]
    means = [m1, m2, m3, m4, m5, m6, m7]
    stanD = [s1, s2, s3, s4, s5, s6, s7]


    plt.errorbar(values, means, stanD, label='exhaustive')

    a = heurist<.05
    h1 = heurist[a]    
    m1 = np.mean(h1)
    s1 = np.std(h1)

    a1 = heurist > .05
    a2 = heurist < .1
    a = a1*a2
    h2 = heurist[a]
    m2 = np.mean(h2)
    s2 = np.std(h2)

    a1 = heurist > .1
    a2 = heurist < .15
    a = a1*a2
    h3 = heurist[a]
    m3 = np.mean(h3)
    s3 = np.std(h3)

    a1 = heurist > .15
    a2 = heurist < .2
    a = a1*a2
    h4 = heurist[a]
    m4 = np.mean(h4)
    s4 = np.std(h4)

    a1 = heurist > .4
    a2 = heurist < .45
    a = a1*a2
    h5 = heurist[a]
    m5 = np.mean(h5)
    s5 = np.std(h5)

    a1 = heurist > .45
    a2 = heurist < .5
    a = a1*a2
    h6 = heurist[a]
    m6 = np.mean(h6)
    s6 = np.std(h6)
    
    a1 = heurist > .5
    a2 = heurist < .55
    a = a1*a2
    h7 = heurist[a]
    m7 = np.mean(h7)
    s7 = np.std(h7)

    values = [0, .05, .1, .15, .4, .45, .5]
    means = [m1, m2, m3, m4, m5, m6, m7]
    stanD = [s1, s2, s3, s4, s5, s6, s7]

    plt.errorbar(values, means, stanD, label='Holiday heuristic')

    plt.legend(loc='lower right')
    plt.show()
    
    return

#this is just a histogram of the error in 
#def errorHistogram(exhaustFile, heuristFile):
    
