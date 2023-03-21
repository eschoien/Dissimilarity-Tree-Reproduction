import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
from matplotlib import pyplot as plt
import random
import numpy
# Run from DTR directory, will not find files if cd'ed into folder of this file

def calculate_precision(data, k):
    precisions = []
    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            precisions.append(1/k)
        else:
            precisions.append(0/k)
        
    return sum(precisions)/len(precisions)

def calculate_recall(data, k):
    recalls = []
    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            recalls.append(1)
        else:
            recalls.append(0)
    return sum(recalls)/len(recalls)

# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v4/'
ks = range(1, 384)
p = 10
thresholds = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
querysets = ["partial", "complete"]
d = 500
# -----------------

cmap = plt.cm.get_cmap('hsv', 400)

for q in querysets:
    for j in thresholds:
        
        
        signature_data = json.load(open(basePath + q + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json')) 
        
        precisions = []
        recalls = []

        for k in ks:
            precision = calculate_precision(signature_data, k)
            recall = calculate_recall(signature_data, k)
            plt.plot(recall, precision, marker='o', color=cmap(k))
            precisions.append(precision)
            recalls.append(recall)
        plt.plot(recalls, precisions, label=False, color='black' if q == 'partial' else 'blue')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc='upper left')
plt.grid()
plt.show()

                    