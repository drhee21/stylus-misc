import matplotlib.pyplot as plt
import pandas as pd


def readResult(file):

    #Input: a csv file. If not in the same folder as the code running it, then include the location too
    #Output: a list of genes, a list of archetypes, and an array of the data
    aData = pd.read_csv(file)
    m = len(aData) - 1 
    n = len(aData.iloc[0]) - 1
    genes = []
    for x in range(m):
        genes.append(aData.iat[x,1])
    headers = aData.columns
    archetypes = []
    for x in headers[2:]:
        archetypes.append(x)
    data = aData[archetypes].to_numpy()
    return genes, archetypes, data

def readScore(file, han_char, gene):

    #Input: a csv file, a han character, and a gene. Keep the ".han" or the ".gene" if they were also there in the table
    #Output: the score on that file from that han character and gene
    aData = pd.read_csv(file)
    m = len(aData) - 1 
    genes = []
    for x in range(m):
        genes.append(aData.iat[x,1])

    x = 0;
    for i in genes:
        if(gene == i):
            break
        x += 1
    if(x==m):
        print("Error, that gene cannot be found")
        data = -1
    else:
        print(x)
        num = aData.loc[x, han_char]
        data = num.astype(float)
        print(data)
    return data