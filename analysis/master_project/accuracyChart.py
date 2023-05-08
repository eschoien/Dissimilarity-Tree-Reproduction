import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


# objects = []
# h_time = []
# v_time = []
# c_time = []


# for i in range(10):
#     if i == 4:
#         obj = f'T{380}'
#         objects.append(obj)
#         h_time.append(sorted_horizontal[obj])
#         v_time.append(sorted_vertical[obj])
#         c_time.append(sorted_combinded[obj])
#         continue
    
#     if i == 7:
#         obj = f'T{275}'
#         objects.append(obj)
#         h_time.append(sorted_horizontal[obj])
#         v_time.append(sorted_vertical[obj])
#         c_time.append(sorted_combinded[obj])
#         continue

#     if i == 5:
#         obj = f'T{373}'
#         objects.append(obj)
#         h_time.append(sorted_horizontal[obj])
#         v_time.append(sorted_vertical[obj])
#         c_time.append(sorted_combinded[obj])
#         continue
        
#     x = random.randint(1, 383)
#     obj = f'T{x}'
#     objects.append(obj)
#     h_time.append(sorted_horizontal[obj])
#     v_time.append(sorted_vertical[obj])
#     c_time.append(sorted_combinded[obj])

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
for p in permutations:
    accuracies[p] = {}
    for queryset in querysets:
        for d in limits:
            accuracies[p][d] = {}
            for j in thresholds:
                # accuracies[p][d][j] = []
                signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+str(j)+'-'+str(d)+'-'+str(p)+'.json'))
                
                accuracies[p][d][j] = calculate_accuracy(signature_data, k)[1] 

data = {
    'Threshold': thresholds,
}
df = pd.DataFrame(data)

for p in permutations:
    dl100 = accuracies[p][limits[0]].values()
    print(dl100)
    dl500 = accuracies[p][limits[1]].values()
    dl1000 = accuracies[p][limits[2]].values()
    df.insert(len(df.columns), f'{p}-Desc_limit-100', dl100)
    df.insert(len(df.columns), f'{p}-Desc_limit-500', dl500)
    df.insert(len(df.columns), f'{p}-Desc_limit-1000', dl1000)
print(df)

N = len(thresholds)
ind = np.arange(N) 
width = 0.25
print(df['10-Desc_limit-100'])

bar1 = [plt.bar(ind, df['10-Desc_limit-100'], width=width),
        plt.bar(ind, df['50-Desc_limit-100'], bottom=df['10-Desc_limit-100'], width=width),
        plt.bar(ind, df['100-Desc_limit-100'], bottom=df['10-Desc_limit-100']+df['50-Desc_limit-100'], width=width)]
bar2 = [plt.bar(ind+width, df['10-Desc_limit-500'], width=width),
        plt.bar(ind+width, df['50-Desc_limit-500'], bottom=df['10-Desc_limit-500'], width=width),
        plt.bar(ind+width, df['100-Desc_limit-500'], bottom=df['10-Desc_limit-500']+df['50-Desc_limit-500'], width=width)]
bar3 = [plt.bar(ind+width*2, df['10-Desc_limit-1000'], width=width),
        plt.bar(ind+width*2, df['50-Desc_limit-1000'], bottom=df['10-Desc_limit-1000'], width=width),
        plt.bar(ind+width*2, df['100-Desc_limit-1000'], bottom=df['10-Desc_limit-1000']+df['50-Desc_limit-1000'], width=width)]

plt.xlabel("Jaccard Thresholds")
plt.ylabel("Accuracy%")
# plt.ylim([0,100])
# plt.xlim([0,1])
# plt.title("test")

plt.xticks(ind+width, thresholds)
plt.legend((bar1[0], bar1[1], bar1[2], bar2[0], bar2[1], bar2[2], bar3[0], bar3[1], bar3[2]), ('P=10 D=100', 'P=50 D=100', 'P=100 D=100', 'P=10 D=500', 'P=50 D=500', 'P=100 D=500', 'P=10 D=1000', 'P=50 D=1000', 'P=100 D=1000'))
plt.show()