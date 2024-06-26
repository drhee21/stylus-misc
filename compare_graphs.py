import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from readCsv import readResult, readScore

#this graphs genes vs scores, with the ability to put multiple on top of each other to clearly see where they differ. I've been using this with the first file
#being the exhaustive scores, and the rest being the heuristics, though unlike in all of the rest of the graphs, it doesn't actually matter for this one
def graphScores(file1, file2 = '', file3 = '', file4 = '', file5 = '', char = -1): 
    #Input: csv files that are archetype/gene score tables (1-4), if multiple, they need to be about the same genes and archetypes 
    g1, a1, d1 = readResult(file1)
    x = g1
    dt1=np.transpose(d1)
    #if you want to graph data from a different archetype, change the char parameter, but to change the default to a different character you can change
    # the 0 to the index of a different archetype
    #I can't graph them all because the csv file I'm using has almost 600 genes, so graphing 600 points alone is a lot, I don't want
    #to also multiply that by 11 archetypes
    if char == -1:
        arche=0
    else:
        arche = char
    y = dt1[arche]

    #Makes and sets the size of the graph, and plots the first set of data. Labels removed for poster
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(x, y, label = file1[:-4])
    '''
    ax.set_ylabel('Scores')
    ax.set_xlabel('Genes in arbitrary order')
    ax.set_title('Score of Heuristics and Exhaustive using ' + a1[arche])
    '''
    #Plots the other files on top of the first one, the if statements are because if you don't use all 5 files then it won't give an error
    if file2:
        g2, a2, d2 = readResult(file2)
        dt2=np.transpose(d2)
        y = dt2[arche]
        ax.plot(x,y, label = file2[:-4])
    if file3:
        g3, a3, d3 = readResult(file3)
        dt3=np.transpose(d3)
        y = dt3[arche]
        ax.plot(x,y, label = file3[:-4])
    if file4:
        g4, a4, d4 = readResult(file4)
        dt4=np.transpose(d4)
        y = dt4[arche]
        ax.plot(x,y, label = file4[:-4])
    if file5:
        g5, a5, d5 = readResult(file5)
        dt5=np.transpose(d5)
        y = dt5[arche]
        ax.plot(x,y, label = file5[:-4])

    #Graph settings stuff, removing the x axis because the gene names would just cover each other, making the y-axis numbers bigger, etc
    ax.set_xticklabels([])
    plt.yticks(fontsize = 20)
    plt.legend(loc='upper left')
    plt.show()
    return

#This graphs the same as the above function, but sorts the scores by their exhaustive score so that similar scores are together
def graphScoresOrdered(exhaustFile, f1 = '', f2 = '', f3 = '', f4 = '', char = -1):
    #Input: The exhaustive csv file, with up to 4 heuristic files. See readResult for info on csv files. char is the index of the archetype column that you
    # want to read if you want to specify or loop through them all

    #Makes the default archetype whichever is first in the csv, this could be edited but it's easier to just imput the archetype number you need
    if char == -1:
        arche=0
    else:
        arche = char
    #this reads all of the csvs, the if statements are so you don't get an error if you don't include 4 heuristics
    eGene, eArch, eData = readResult(exhaustFile)
    if f1:
        h1Gene, h1Arch, h1Data = readResult(f1)
    if f2:
        h2Gene, h2Arch, h2Data = readResult(f2)
    if f3:
        h3Gene, h3Arch, h3Data = readResult(f3)
    if f4:
        h4Gene, h4Arch, h4Data = readResult(f4)

    #Arranges the data into a 1d array holding all of the scores for the archetype we're using for each file
    dte = np.transpose(eData)
    exhaust = dte[arche]
    order = sorted(exhaust)
    if f1:
        dt1 = np.transpose(h1Data)
        heurist1 = dt1[arche]
    if f2:
        dt2 = np.transpose(h2Data)
        heurist2 = dt2[arche]
    if f3:
        dt3 = np.transpose(h3Data)
        heurist3 = dt3[arche]
    if f4:
        dt4 = np.transpose(h4Data)
        heurist4 = dt4[arche]

    #The below code sorts the exhaustive scores in order, but more importantly makes a list of the indexes in score-increasing order so that the heuristics
    # can be graphed in that order as well
    index = []
    dSort = []
    min = 0

    while dSort != order:
        minscore = 1
        tempindex = -1
        for i in range(len(exhaust)):
            if exhaust[i] < minscore and exhaust[i] > min:
                minscore = exhaust[i]
                tempindex = i
        min = minscore
        index.append(tempindex)
        dSort.append(exhaust[tempindex])

    #Setting graph size and plotting data. Commented out plots were for the poster
    plt.figure(figsize = (15, 5))
    #plt.figure(figsize = (15, 10))
    plt.plot(order, label = 'Exhaustive', color = 'purple')

    if f1:
        h1Sort = heurist1[index]
        plt.plot(h1Sort, label = f1[:-4])
        #plt.plot(h1Sort, label = 'Heuristic 1', color = 'cornflowerblue')
    if f2:
        h2Sort = heurist2[index]
        #plt.plot(h2Sort, label = 'Heuristic 2', color = 'darkorange')
        plt.plot(h2Sort, label = f2[:-4])
    if f3:
        h3Sort = heurist3[index]
        #plt.plot(h3Sort, label = 'Heuristic 3', color = 'forestgreen')
        plt.plot(h3Sort, label = f3[:-4])
    if f4:
        h4Sort = heurist4[index]
        #plt.plot(h4Sort, label = 'Greedy Heuristic', color = 'crimson')
        plt.plot(h4Sort, label = f4[:-4])
        
    plt.legend(loc='upper left')
    #plt.legend(fontsize = 20)
    plt.xticks([])
    #plt.yticks(fontsize = 20)
    plt.show()
    return 

#While the previous two graphing functions only graph the scores generated from one archetype, this iterates through all of the scores, and also sorts them
# by exaustive score
def graphScoresOAll(exhaustFile, file1 = '', file2 = '', file3 = '', file4 = '', zoom = -1):
    #Input: An exhaustive csv file, and up to 4 heuristic score csv files. Zoom removes the data with exhaustive scores greater than zoom from your graph so 
    # you can more easily see the smaller values. I highly recommend using it with this graph because since there are so many data points, it is rather 
    # difficult to see anything useful with the entire graph

    #reading and preparing exhaustive data
    eGene, eArch, eData = readResult(exhaustFile)
    exhaust = eData.flatten()
    order = sorted(exhaust)
    fig, ax = plt.subplots(figsize=(30,5))

    #zoom default is the greatest exhaustive score so everything will be graphed
    if zoom == -1:
        zoom = max(exhaust)
    '''
    ax.set_ylabel('Scores')
    ax.set_xlabel('Genes in arbitrary order')
    ax.set_title('Score of Heuristics and Exhaustive')
    '''

    #Creating the index list for exhaustive scores in ascending order
    index = []
    dSort = []
    min = 0

    while dSort != order:
        minscore = 1
        tempindex = -1
        for i in range(len(exhaust)):
            if exhaust[i] < minscore and exhaust[i] > min:
                minscore = exhaust[i]
                tempindex = i
        min = minscore
        if min >= zoom:
            break
        index.append(tempindex)
        dSort.append(exhaust[tempindex])

    #reading the heuristic files if they're given and plotting everything
    ax.plot(dSort)
    
    if file1:
        g1, a1, d1 = readResult(file1)
        heurist1 = d1.flatten()
        h1Sort = heurist1[index]
        ax.plot(h1Sort, label = file1[:-4])
    if file2:
        g2, a2, d2 = readResult(file2)
        heurist2 = d2.flatten()
        h2Sort = heurist2[index]
        ax.plot(h2Sort, label = file2[:-4])
    if file3:
        g3, a3, d3 = readResult(file3)
        heurist3 = d3.flatten()
        h3Sort = heurist3[index]
        ax.plot(h3Sort, label = file3[:-4])
    if file4:
        g4, a4, d4 = readResult(file4)
        heurist4 = d4.flatten()
        h4Sort = heurist4[index]
        ax.plot(h4Sort, label = file4[:-4])

    #graph settings
    ax.set_xticklabels([])
    #plt.legend(loc='upper right')
    plt.yticks(fontsize = 20)
    plt.show()

    return


#This graph compares heuristic scores to exhaustive scores, a perfect heuristic would be a straight "45 degree" line
def graphExhaustHeurist(exhaustFile, heuristFile, zoom = -1):
    #Input: the exhaustive score csv file and the heuristic score csv file. These are the files generated by my GeneArchetype_Heuristic_Table or the exhaustive one, or a table of similar format. The zoom input is a number so that you can zoom in the graph if you so desire to focus more on the smaller numbers that are more likely incorrect instead of looking at all of the data provided

    #reading files
    eGene, eArch, eData = readResult(exhaustFile)
    hGene, hArch, hData = readResult(heuristFile)

    #making data 1d instead of 2d
    exhaust = eData.flatten()
    heurist = hData.flatten()

    #zoom functionality removes data relating to exhaustive scores greater than zoom
    if zoom!=-1:
        x = exhaust < zoom
        exhaust = exhaust[x]
        heurist = heurist[x]

    #plotting and graph settings (labels removed for sake of the poster, but they work)
    plt.scatter(exhaust, heurist, s=.5)
    '''
    plt.xlabel('Exhaustive Scores')
    plt.ylabel(heuristFile[:-4] + ' Scores')
    plt.title('Exhaustive vs Heuristic Scores')
    '''
    plt.show()
    return
    
#This creates a plot of the error (if e = ehaustive score and h = heuristic score, then error = (e-h)/h) of the heuristics sorted by their exhaustive scores
#High numbers are bad heuristics, low numbers are good ones
#These graphs are all put on one graph, so it's easier to compare.
def percentError(exhaustFile, f1, f2 = '', f3 = '', f4 = '', zoom = -1):
    #Input: a csv of exhaustive scores, and at least 2 csvs of heuristic files. See readResult for more info on the csv format required. Zoom ignores data
    # greater than itself to allow focus on smaller numbers

    #reading all files that are included
    eGene, eArch, eData = readResult(exhaustFile)
    h1Gene, h1Arch, h1Data = readResult(f1)
    if f2:
        h2Gene, h2Arch, h2Data = readResult(f2)
    if f3:
        h3Gene, h3Arch, h3Data = readResult(f3) 
    if f4:
        h4Gene, h4Arch, h4Data = readResult(f4) 

    #makes data 1d and prepares arrays to hold data
    exhaust = eData.flatten()
    heurist1 = h1Data.flatten()
    if f2:
        heurist2 = h2Data.flatten()
        mean2 = []
        stanD2 = []
    if f3:
        heurist3 = h3Data.flatten()
        mean3 = []
        stanD3 = []
    if f4:
        heurist4 = h4Data.flatten()
        mean4 = []
        stanD4 = []
    if zoom == -1:
        zoom = max(exhaust)

    num = zoom/25
    
    values = []
    mean1 = []
    stanD1 = []

    #this array finds the mean and standard deviation for the error of each heuristic when the data is broken up into 25 different bins so long as there is at
    # least one data point in that range.
    for j in range(0,25):
        #creates a boolean list (a) where true means that the data falls in the desired range and false means outside. this can be used to create a version 
        # of each data set only including the desired values
        a1 = exhaust > (j*.005)
        a2 = exhaust < ((j+1)*.005)
        a = a1*a2
         
        e = exhaust[a]
        h1 = heurist1[a]
        err1 = []
        if f3:
            h2 = heurist2[a]
            err2 = []
        if f3:
            h3 = heurist3[a]
            err3 = []
        if f4:
            h4 = heurist4[a]
            err4 = []
        #calculates the error
        if len(e) != 0:
            values.append((num*j)+(num/2))
            for i in range(len(e)):
                x = (e[i]-h1[i])/e[i]
                err1.append(x)
                if f2:
                    x = (e[i]-h2[i])/e[i]
                    err2.append(x)
                if f3:
                    x = (e[i]-h3[i])/e[i]
                    err3.append(x)
                if f4:
                    x = (e[i]-h4[i])/e[i]
                    err4.append(x)
            #adding data to relevant array
            mean1.append(np.mean(err1))
            stanD1.append(np.std(err1))
            if f2:
                mean2.append(np.mean(err2))
                stanD2.append(np.std(err2))
            if f3:
                mean3.append(np.mean(err3))
                stanD3.append(np.std(err3))
            if f4:
                mean4.append(np.mean(err4))
                stanD4.append(np.std(err4))

    print("Note: the size of the error bars represents the standard deviation, but the bars are centered around the median on the graph. There aren't any negative errors even if the bars show it")
    #Creating graph size and plotting the data, commented out plots were for the poster
    fig, ax = plt.subplots(figsize=(25, 5))
    #fig, ax = plt.subplots(figsize=(15, 10))
    ax.errorbar(values, mean1, stanD1, label = f1[:-4], fmt = '*', elinewidth = .3, markersize = 4)
    #ax.errorbar(values, mean1, stanD1, label = 'Heuristic 1', fmt = '*', elinewidth = .3, markersize = 10)
    if f2:
        ax.errorbar(values, mean2, stanD2, label = f2[:-4], fmt = '*', elinewidth = .3, markersize = 4)
        #ax.errorbar(values, mean2, stanD2, label = 'Heuristic 2', fmt = '*', elinewidth = .3, markersize = 10)
    if f3:
        ax.errorbar(values, mean3, stanD3, label = f3[:-4], fmt = '*', elinewidth = .3, markersize = 4)
        #ax.errorbar(values, mean3, stanD3, label = 'Heuristic 3', fmt = '*', elinewidth = .3, markersize = 10)
    if f4:
        ax.errorbar(values, mean4, stanD4, label = f4[:-4], fmt = '*', elinewidth = .3, markersize = 4)
        #ax.errorbar(values, mean4, stanD4, label = 'Greedy Heuristic', fmt = '*', elinewidth = .3, markersize = 10)

    #graph settings, labels once again removed for the poster but they are functional
    '''
    ax.set_title('Error for different heuristics, with standard dev error bars')
    ax.set_xlabel('Correct score for each gene')
    ax.set_ylabel('Error ((exhaustive-heuristic)/exhaustive)')
    '''
    #plt.legend(loc='upper right')
    #plt.yticks(fontsize = 20)
    #plt.xticks(fontsize = 20)
    plt.grid()
    plt.show()
    return
#this graphs the percent of incorrect values for each heuristic as a bar graph. This is different than the error ones because error gave you some credit if the
# heuristic found a pretty good score, but not the best one. This only counts those that have the entirely correct score and disregards any that fail
#This will make a gap over any data that it doesn't have
def percentMismatchBar(exhaustFile, f1, f2 = '', f3 = '', zoom = -1):
    #Input: exhaustive scores csv, heuristic scores csv (see readResult input), zoom allows someone to change the scale of the graph to be smaller, so it will 
    # display more information about just the smaller numbers present.

    #reading the files
    eGene, eArch, eData = readResult(exhaustFile)
    h1Gene, h1Arch, h1Data = readResult(f1)
    if f2:
        h2Gene, h2Arch, h2Data = readResult(f2)
    if f3:
        h3Gene, h3Arch, h3Data = readResult(f3)

    #making the files 1d
    exhaust = eData.flatten()
    heurist1 = h1Data.flatten()
    if f2:
        heurist2 = h2Data.flatten()
    if f3:
        heurist3 = h3Data.flatten()
    
    #setting default zoom
    if zoom == -1:
        zoom = max(exhaust)

    #setting up data for loop
    num = zoom/25
    x = []
    percents1 = []
    if f2:
        percents2 = []
    if f3:
        percents3 = []

    #creates 25 bins of equal spacing and finds the percent mismatch in each
    for i in range(0,25):
        #filtering data to only have the desired range
        a1 = exhaust > (num*i)
        a2 = exhaust < (num*(i+1))
        a = a1*a2
        e = exhaust[a]
        h1 = heurist1[a]
        if f2:
            h2 = heurist2[a]
        if f3:
            h3 = heurist3[a]
        #If there is data in the range, sum the incorrect data. find the percentage, and add it to proper list
        if len(e) > 0:
            x.append((num*i)+(num/2))
            count1 = 0
            count2 = 0
            count3 = 0
            for j in range(len(e)):
                if e[j] != h1[j]:
                    count1+= 1
                if f2 and e[j] != h2[j]:
                    count2+= 1
                if f3 and e[j] != h3[j]:
                    count3+= 1
            p1 = count1/len(e) * 100
            percents1.append(p1)
            if f2:
                p2 = count2/len(e)*100
                percents2.append(p2)
            if f3:
                p3 = count3/len(e) *100
                percents3.append(p3)
        #If there isn't any data, make that bar be blank
        else:
            x.append(-1)
            percents1.append(0)
            if f2:
                percents2.append(0)
            if f3:
                percents3.append(0)

    #This sets up the offset of the bars so they aren't on top of each other
    width = .3
    b0 = np.arange(len(percents1)) 
    b1 = [x + width for x in b0] 

    if f2 or f3:
        b1 = np.arange(len(percents1)) 
        b2 = [x + width for x in b1] 
        b3 = [x + width for x in b2] 

    #Graph labels are removed for the poster
    '''
    plt.xlabel('Scores')
    plt.ylabel('Percents')
    plt.title('Percent Correct of Heuristic(s) ')
    '''
    #Because of the way that bar graphs work, the titles need to all be strings. Also, the bar graphs without data need to be blank
    for a in range(len(x)):
        l = round(x[a], 4)
        x[a] = str(l)
        if x[a] == '-1':
            x[a] = ''

    #Creating graph size and plotting data, colors are set to match with the default colors for the other graphs
    fig = plt.figure(figsize = (20, 5))
        
    plt.bar(b1, percents1, color = 'royalblue', width = width)
    if f2:
        plt.bar(b2, percents2, color = 'darkorange', width = width)
    if f3:
        plt.bar(b3, percents3, color = 'g', width = width)

    #Graph settings
    #plt.legend(loc='lower right')
    plt.xticks([r + width for r in range(len(percents1))], 
        x)
    plt.grid(axis='y')
    plt.show()

    return

#This has the same data as the above function, but plots it instead of using a bar graph. This makes it easier to compare to percentError. It creates a graph
# of the percent of data in a subset of the data that is a mismatch (if the heuristic score isn't the same as the exhaustive one, it's a mismatch) for 
# 25 subsets and graphs them, allowing for up to 4 heuristics. This can handle one more heuristic than the above function simply because it's easier
# to add another data point than to adjust the bars to make room for another one
def percentMismatchPlot(exhaustFile, f1, f2 = '', f3 = '', f4 = '', zoom = -1):

    #Input: exhaustive scores csv, heuristic scores csv (see readResult input), zoom allows someone to change the scale of the graph to be smaller, so it will 
    # display more information about just the smaller numbers present.

    #read csv files
    eGene, eArch, eData = readResult(exhaustFile)
    h1Gene, h1Arch, h1Data = readResult(f1)
    if f2:
        h2Gene, h2Arch, h2Data = readResult(f2)
    if f3:
        h3Gene, h3Arch, h3Data = readResult(f3)
    if f4:
        h4Gene, h4Arch, h4Data = readResult(f4)

    #make data 1d
    exhaust = eData.flatten()
    heurist1 = h1Data.flatten()
    if f2:
        heurist2 = h2Data.flatten()
    if f3:
        heurist3 = h3Data.flatten()
    if f4:
        heurist4 = h4Data.flatten()
    
    #zoom default is the max value
    if zoom == -1:
        zoom = max(exhaust)

    #setting up for calculating the percents
    num = zoom/25
    x = []
    percents1 = []
    if f2:
        percents2 = []
    if f3:
        percents3 = []
    if f4:
        percents4 = []

    for i in range(0,25):
        #filtering data to find scores in the desired range
        a1 = exhaust > (num*i)
        a2 = exhaust < (num*(i+1))
        a = a1*a2
        e = exhaust[a]
        h1 = heurist1[a]
        if f2:
            h2 = heurist2[a]
        if f3:
            h3 = heurist3[a]
        if f4:
            h4 = heurist4[a]
        #if there's actually data in that range, count the mismatches, find the percents and add them to the proper list
        if len(e) > 0:
            x.append((num*i)+(num/2))
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0
            for j in range(len(e)):
                if e[j] != h1[j]:
                    count1+= 1
                if f2 and e[j] != h2[j]:
                    count2+= 1
                if f3 and e[j] != h3[j]:
                    count3+= 1
                if f4 and e[j] != h4[j]:
                    count4+= 1
            p1 = count1/len(e) * 100
            percents1.append(p1)
            if f2:
                p2 = count2/len(e)*100
                percents2.append(p2)
            if f3:
                p3 = count3/len(e) *100
                percents3.append(p3)
            if f4:
                p4 = count4/len(e)*100
                percents4.append(p4)

    #labels removed for poster
    '''
    plt.xlabel('Scores')
    plt.ylabel('Percents')
    plt.title('Percent Correct of Heuristic(s) ')
    '''

    #Setting graph size and plotting data
    fig = plt.figure(figsize = (20, 5))
        
    plt.plot(x, percents1, '*')
    if f2:
        plt.plot(x, percents2, '*')
    if f3:
        plt.plot(x, percents3, '*')
    if f4:
        plt.plot(x, percents4, '*')

    #Graph settings
    #plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return