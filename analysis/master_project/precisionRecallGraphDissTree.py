import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
from matplotlib import pyplot as plt
import random
import numpy
# Run from DTR directory, will not find files if cd'ed into folder of this file

def calculate_precision(data, k):
    precisions = []
    
    for queries in data['results']:
        queryFile = queries['queryFile'].split("/")[2]
        exists = False      
        for query in queries['searchResults'][:k]:
            guessedFile = query['objectFilePath'].split("/")[3]
            if queryFile == guessedFile:
                exists = True
        if exists:
            precisions.append(1/k)
        else:
            precisions.append(0/k)
        
    return sum(precisions)/len(precisions)

def calculate_recall(data, k):
    recalls = []
    for queries in data['results']:
        queryFile = queries['queryFile'].split("/")[2]
        exists = False
        for query in queries['searchResults'][:k]:
            guessedFile = query['objectFilePath'].split("/")[3]
            if queryFile == guessedFile:
                exists = True
        if exists:
            recalls.append(1)
        else:
            recalls.append(0)
    return sum(recalls)/len(recalls)

# --- ARGUMENTS ---
basePath = 'output/dissTree/measurements/v4/'
ks = range(1, 384)
# -----------------

cmap = plt.cm.get_cmap('hsv', 400)

        
        
signature_data = json.load(open(basePath + 'measurement-383.json')) 

precisions = []
recalls = []

for k in ks:
    precision = calculate_precision(signature_data, k)
    recall = calculate_recall(signature_data, k)
    plt.plot(recall, precision, marker='o', color=cmap(k))
    precisions.append(precision)
    recalls.append(recall)
    
plt.plot(recalls, precisions, color='black')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc='upper left')
plt.grid()
plt.show()

                    