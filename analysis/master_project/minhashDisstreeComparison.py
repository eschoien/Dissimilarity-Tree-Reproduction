import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random

def calculate_accuracy(data, k):

    correct_guesses = 0

    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            correct_guesses += 1

    amount = str(correct_guesses) + '/' + str(len(data['results']))
    accuracy = (correct_guesses / len(data['results'])) * 100
    return amount, accuracy


def calculate_time(data):
    time = 0.0
    for query in data['results']:
        time += query['executionTimeSeconds']

    avg_time = time / len(data['results'])
    
    return time, avg_time

def calculate_accuracyDiss(data, k, q):

    correct_guesses = 0

    for queries in data['results']:
        if q == 'complete':
            queryFile = queries['queryFile'].split("/")[3]
        else:
            queryFile = queries['queryFile'].split("/")[2]
        for query in queries['searchResults'][:k]:
            guessedFile = query['objectFilePath'].split("/")[3]
            if queryFile == guessedFile:
                correct_guesses += 1

    amount = str(correct_guesses) + '/' + str(len(data['results']))
    accuracy = (correct_guesses / len(data['results'])) * 100
    return amount, accuracy

# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v4/'
hashPath = 'output/hashtableMeasurements/v8/'
k = 1
p = 10
thresholds =  ["0.3","0.4"]
# thresholds =  [0.2,0.3,0.4,0.5]
querysets = ["partial", "complete"]
limit = 100
# -----------------
accuracies = {}
times = {}

for queryset in querysets:
    dissTree_data = json.load(open('output/dissTree/measurements/v4/' + queryset + '_objects/measurement-383.json'))
    accuracies[queryset] = []
    times[queryset] = []
    hashd = queryset + 'H'
    accuracies[hashd] = []
    times[hashd] = []
    for j in thresholds:
        signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(limit)+'-'+str(p)+'.json'))
        hash_data = json.load(open(hashPath + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(limit)+'-'+str(p)+'.json'))
        
        accuracies[queryset].append(calculate_accuracy(signature_data, k)[1])
        times[queryset].append(calculate_time(signature_data)[1])

        accuracies[hashd].append(calculate_accuracy(hash_data, k)[1])
        times[hashd].append(calculate_time(hash_data)[1])

#     accuracies[queryset].append(calculate_accuracyDiss(dissTree_data, k, queryset)[1])
#     times[queryset].append(calculate_time(dissTree_data)[1])

# thresholds.append('diss tree')

data = {
    'Threshold': thresholds,
    'AccuraciesP': accuracies['partial'],
    'TimesP': times['partial'],
    'AccuraciesC': accuracies['complete'],
    'TimesC': times['complete'],
    'AccuraciesHP': accuracies['partialH'],
    'TimesHP': times['partialH'],
    'AccuraciesHC': accuracies['completeH'],
    'TimesHC': times['completeH']
}
df = pd.DataFrame(data)

N = len(thresholds)
ind = np.arange(N) 
width = 0.3

os.makedirs('output/project-results/hashtable', exist_ok=True)

print(df['AccuraciesP'])
print(df['AccuraciesC'])
print(df['TimesP'])
print(df['TimesC'])

plt.figure(1)
bar1 = [plt.bar(ind, df['AccuraciesP'], width=width, label='Minhash', color='r')]
bar2 = [plt.bar(ind+width, df['AccuraciesHP'], width=width, label='Hash Table', color='g')]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Accuracy%")
plt.ylim([0,100])

plt.xticks(ind+(width/2), thresholds)
plt.legend()
# plt.show()
plt.savefig('output/project-results/hashtable/partialacc_comparison.png')

plt.figure(2)
bar1 = [plt.bar(ind, df['AccuraciesC'], width=width, label='Minhash', color='r')]
bar2 = [plt.bar(ind+width, df['AccuraciesHC'], width=width, label='Hash Table', color='g')]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Accuracy%")
plt.ylim([0,100])

plt.xticks(ind+(width/2), thresholds)
plt.legend()
# plt.show()
plt.savefig('output/project-results/hashtable/completeacc_comparison.png')

plt.figure(3)
bar1 = [plt.bar(ind, df['TimesP'], width=width, label='Minhash', color='r')]
bar2 = [plt.bar(ind+width, df['TimesHP'], width=width, label='Hash Table', color='g')]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Time (s)")

plt.xticks(ind+(width/2), thresholds)
plt.legend()
# plt.show()
plt.savefig('output/project-results/hashtable/partialtime_comparison.png')

plt.figure(4)
bar1 = [plt.bar(ind, df['TimesC'], width=width, label='Minhash', color='r')]
bar2 = [plt.bar(ind+width, df['TimesHC'], width=width, label='Hash Table', color='g')]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Time (s)")

plt.xticks(ind+(width/2), thresholds)
plt.legend()
# plt.show()
plt.savefig('output/project-results/hashtable/completetime_comparison.png')