import json
from scripts.prettytable import PrettyTable
from datetime import timedelta
# Run from DTR directory, will not find files if cd'ed into folder of this file


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
# basePath = 'output/lsh/measurements/v4/'
basePath = 'output/hashtableMeasurements/v8/'
ks = [1]
permutations = [50]
thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
querysets = ["partial", "complete"]
limits = [500]
# -----------------


for k in ks:

    for p in permutations:

        outputTable = PrettyTable()
        outputTable.align = "r"
        outputTable.add_column("Jaccard", thresholds+["Total time"])
        totalTime = 0

        for queryset in querysets:

            outputTable.add_column("-", ["-"]*11,)

            for d in limits:

                columnAccuracy = []
                columnTimes = []

                for j in thresholds:

                    try:
                        signature_data = json.load(open(basePath + queryset + '_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))

                        signature_accuracy = calculate_accuracy(signature_data, k)
                        columnAccuracy.append('{:.2f}%'.format(round(signature_accuracy[1],2)).rjust(7, " "))
                        #columnAccuracy.append(f'{round(signature_accuracy[1],2)}%, {round(time, 3)}') # Also print avg execution time
                        #columnAccuracy.append(f'{signature_accuracy[0]}') # Print number of correct results / number of queries

                        time, avg_time = calculate_time(signature_data)
                        columnTimes.append(time)
                        totalTime += time
                    
                    except:
                        columnAccuracy.append('-')
                        columnTimes.append(0)
                        
                outputTable.add_column(queryset[0].upper()+str(d), columnAccuracy+[timedelta(seconds=round(sum(columnTimes)))])

        print(str("{"+"Partial".center(29,"-")+"}   {"+"Complete".center(29,"-")+"}").rjust(82, " "))
        print(outputTable)
        print(f'Results for top-k={k}, Permutations p={p}, Total time: {timedelta(seconds=round(totalTime))}')
        print()
        print()