import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
from matplotlib import pyplot as plt
import random
import numpy
# Run from DTR directory, will not find files if cd'ed into folder of this file


def calculate_accuracy(data, k):

    correct_guesses = 0

    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            correct_guesses += 1

    amount = str(correct_guesses) + '/' + str(len(data['results']))
    accuracy = (correct_guesses / len(data['results'])) * 100
    return amount, accuracy

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v3/'
ks = [1, 3, 5, 10, 100, 200, 300, 383]
ks = range(1, 384)
permutations = [50]
thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
querysets = ["partial"]
limits = [500]
# -----------------
accuracies = {}

cmap = get_cmap(len(thresholds)*len(permutations)+3)
for p in permutations:
    for queryset in querysets:
        for d in limits:
            for j in thresholds:
                accuracies[j] = []
                signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))
                for k in ks:
                    accuracies[j].append(calculate_accuracy(signature_data, k)[1])

                plt.plot(ks, accuracies[j], label=f'{j}', c=cmap((limits.index(d)*len(thresholds))+thresholds.index(j)))
plt.legend(loc="lower right")
plt.show()

                    