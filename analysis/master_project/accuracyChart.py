import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v3/'
k = 1
permutations = [10, 50, 100]
# thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
thresholds =  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
querysets = ["partial"]
limits = [100, 500, 1000]
# -----------------
accuracies = {}
times= {}
for p in permutations:
    accuracies[p] = {}
    times[p] = {}
    for queryset in querysets:
        for d in limits:
            accuracies[p][d] = {}
            times[p][d] = {}
            for j in thresholds:
                # accuracies[p][d][j] = []
                signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+str(j)+'-'+str(d)+'-'+str(p)+'.json'))
                
                accuracies[p][d][j] = calculate_accuracy(signature_data, k)[1]
                times[p][d][j] = calculate_time(signature_data)[0]

data = {
    'Threshold': thresholds,
}
df = pd.DataFrame(data)

for p in permutations:
    dl100 = accuracies[p][limits[0]].values()
    dl500 = accuracies[p][limits[1]].values()
    dl1000 = accuracies[p][limits[2]].values()
    df.insert(len(df.columns), f'{p}-Desc_limit-100', dl100)
    df.insert(len(df.columns), f'{p}-Desc_limit-500', dl500)
    df.insert(len(df.columns), f'{p}-Desc_limit-1000', dl1000)

    t100 = times[p][limits[0]].values()
    t500 = times[p][limits[1]].values()
    t1000 = times[p][limits[2]].values()
    df.insert(len(df.columns), f'{p}-Desc_limit-100-Time', t100)
    df.insert(len(df.columns), f'{p}-Desc_limit-500-Time', t500)
    df.insert(len(df.columns), f'{p}-Desc_limit-1000-Time', t1000)


print(df)

N = len(thresholds)
ind = np.arange(N) 
width = 0.25

plt.figure(1)
bar1 = [plt.bar(ind, df['10-Desc_limit-100'], width=width, label='P=10 D=100', color='c')
        ]
bar2 = [plt.bar(ind+width, df['10-Desc_limit-500'], width=width, label='P=10 D=500', color='b')
        ]
bar3 = [plt.bar(ind+width*2, df['10-Desc_limit-1000'], width=width, label='P=10 D=1000', color='darkorange')
        ]


plt.xlabel("Jaccard Thresholds")
plt.ylabel("Accuracy%")
plt.ylim([0,100])

plt.xticks(ind+width, thresholds)
plt.legend()
plt.savefig('output/project-results/minhash/accuracy_comparison.png')

plt.figure(2)
bar1 = [plt.bar(ind, df['10-Desc_limit-100-Time'], width=width, label='P=10 D=100 Time', color='c'),
        ]
bar2 = [plt.bar(ind+width, df['10-Desc_limit-500-Time'], width=width, label='P=10 D=500 Time', color='b'),
        ]
bar3 = [plt.bar(ind+width*2, df['10-Desc_limit-1000-Time'], width=width, label='P=10 D=1000 Time', color='darkorange'),
        ]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Time (s)")

plt.xticks(ind+width, thresholds)
plt.legend()
plt.savefig('output/project-results/minhash/time_comparison.png')

plt.figure(3)
bar1 = [plt.bar(ind, df['10-Desc_limit-500'], width=width, label='P=10 D=500', color='r')
        ]
bar2 = [plt.bar(ind+width, df['50-Desc_limit-500'], width=width, label='P=50 D=500', color='b')
        ]
bar3 = [plt.bar(ind+width*2, df['100-Desc_limit-500'], width=width, label='P=100 D=500', color='g')
        ]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Accuracy%")
plt.ylim([0,100])

plt.xticks(ind+width, thresholds)
plt.legend()
plt.savefig('output/project-results/minhash/accuracy_comparisonP.png')

plt.figure(4)
bar1 = [plt.bar(ind, df['10-Desc_limit-500-Time'], width=width, label='P=10 D=500 Time', color='r'),
        ]
bar2 = [plt.bar(ind+width, df['50-Desc_limit-500-Time'], width=width, label='P=50 D=500 Time', color='b'),
        ]
bar3 = [plt.bar(ind+width*2, df['100-Desc_limit-500-Time'], width=width, label='P=100 D=500 Time', color='g'),
        ]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Time (s)")

plt.xticks(ind+width, thresholds)
plt.legend()
plt.savefig('output/project-results/minhash/time_comparisonP.png')