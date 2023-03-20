import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import random
import numpy
# Run from DTR directory, will not find files if cd'ed into folder of this file

def avg_precision(data, k):
    precisions = []
    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            precisions.append(1/k)
        else:
            precisions.append(0/k)
    avg_precision = sum(precisions)/ len(precisions)
    return avg_precision


def avg_recall(data, k):
    recalls = []
    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            recalls.append(1)
        else:
            recalls.append(0)
    avg_recall = sum(recalls) / len(data['results'])
    return avg_recall

# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v4/'
ks = range(1, 384)
permutations = [10] # [50, 100]
thresholds = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
querysets = ['partial', 'complete']
limits = [500] # [100, 1000]
# -----------------

precisions = {}
recalls = {}

cmap = get_cmap('BrBG', len(thresholds)+1+len(thresholds))

### Experimented with other colormaps:
#cmap = {}
#cmap['partial'] = get_cmap('Reds', len(thresholds))
#cmap['complete'] = get_cmap('Blues', len(thresholds))
#cmap = get_cmap('tab20', len(querysets)*len(thresholds))

for p in permutations:
    for queryset in querysets:
        for d in limits:
            for j in thresholds:
                precisions[j] = []
                recalls[j] = []
                signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))
                for k in ks:
                    precisions[j].append(avg_precision(signature_data, k))
                    recalls[j].append(avg_recall(signature_data, k))

                # Maps the queryset and jaccard threshold to a divering colormap as below:
                #  green gradient         brown gradient
                # P-1.0 ... P-0.1 netrual C-0.1 ... C-1.0

                # The colormap is two times the number of thresholds, as there are two querysets, plus a netrual middle value
                # Line below basically starts in the middle, and either adds or subtracts the jaccard threshold index, depending on queryset
                color_n = len(thresholds) + (thresholds.index(j)+1) * (1 if queryset == 'partial' else -1)
                
                plt.scatter(recalls[j], precisions[j], label=f'{queryset[0].upper()}-{j}', color=cmap(color_n))
                plt.plot(recalls[j], precisions[j], color=cmap(color_n))
plt.legend(loc='upper left')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()
plt.show()

                    