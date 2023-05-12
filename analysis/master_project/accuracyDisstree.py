import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
# Run from DTR directory, will not find files if cd'ed into folder of this file


def calculate_accuracy(data, k, q):

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


def calculate_time(data):
    time = 0.0
    for query in data['results']:
        time += query['executionTimeSeconds']

    avg_time = time / len(data['results'])
    
    return time, avg_time


# --- ARGUMENTS ---
basePath = 'output/dissTree/measurements/v4/'
ks = [1, 3, 5, 10]
# -----------------

outputTable = PrettyTable()
outputTable.align = "r"
outputTable.add_column("Top-K", ks)
querysets = ['partial', 'complete']
totalTime = 0
for queryset in querysets:
    dissTree_data = json.load(open(basePath + queryset + '_objects/measurement-383.json'))
    columnAccuracy = []
    columnTimes = []
    columnAvgTimes = []
    for k in ks:
        try:
            # dissTree_data = json.load(open(basePath + 'measurement-' + str(k) + '.json'))
            dissTree_accuracy = calculate_accuracy(dissTree_data, k, queryset)
            columnAccuracy.append('{:.2f}%'.format(round(dissTree_accuracy[1],2)).rjust(7, " "))
            # print("test")
            #columnAccuracy.append(f'{round(dissTree_accuracy[1],2)}%, {round(time, 3)}') # Also print avg execution time
            #columnAccuracy.append(f'{dissTree_accuracy[0]}') # Print number of correct results / number of queries

            time, avg_time = calculate_time(dissTree_data)
            columnTimes.append(timedelta(seconds=round(time)))
            columnAvgTimes.append(timedelta(seconds=round(avg_time)))
            totalTime += time
        
        except:
            columnAccuracy.append('-')
            columnTimes.append(0)
            columnAvgTimes.append(0)
                        
    outputTable.add_column(f"{queryset} accuracy", columnAccuracy)
    outputTable.add_column("total time", columnTimes)
    outputTable.add_column("average time", columnAvgTimes)


print(outputTable)
print(f'Total time: {timedelta(seconds=round(totalTime))}')
print()
print()