import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
# Run from DTR directory, will not find files if cd'ed into folder of this file


def calculate_time(data):
    time = 0.0
    for query in data['results']:
        time += query['executionTimeSeconds']

    avg_time = time / len(data['results'])
    
    return time, avg_time



# --- ARGUMENTS ---
basePathBreak = 'output/lsh/measurements/v4/'
basePathNoBreak = 'output/lsh/measurements/v_no_break/'
#ks = [1, 3, 5, 10]
permutations = [10, 50, 100]
thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0",]
querysets = ["partial", "complete"]
limits = [100, 500, 1000]
# -----------------

for p in permutations:
    totalTime = 0
    for queryset in querysets:

        speedups = []
        for d in limits:
                columnTimes = []

                for j in thresholds:

                        signature_dataBreak = json.load(open(basePathBreak + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))

                        signature_dataNoBreak = json.load(open(basePathNoBreak + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))

                        timeBreak, avg_timeBreak = calculate_time(signature_dataBreak)
                        timeNoBreak, avg_timeNoBreak = calculate_time(signature_dataNoBreak)

                        speedup = timeNoBreak / timeBreak

                        print(speedup, "\t", queryset, p, d, j)
                        speedups.append(float(speedup))

        print((sum(speedups)/len(speedups))**-1)