import json
import os
from scripts.prettytable import PrettyTable
from datetime import timedelta

# Run from DTR directory, will not find files if cd'ed into folder of this file


def calculate_accuracy(data):

    correct_guesses = 0
    total_guesses = 0

    for i in data['results']:
        total_guesses += 1
        for j in i['bestMatches']:
            if i['queryFileID'] == j:
                correct_guesses += 1
                break

    amount = f'{correct_guesses}/{total_guesses}'
    accuracy = (correct_guesses / total_guesses) * 100
    return amount, accuracy

def calculate_time(data):
    time = 0.0
    for i in data['results']:
        time += i['executionTimeSeconds']

    avg_time = time / len(data['results'])
    
    return time, avg_time

permutations = [10, 50, 100]
thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
limits = [100, 500, 1000, 0, 101, 501, 1001]


for p in permutations:
    timeSec = 0.0
    outputTable = PrettyTable(['Partial', "P100", "P500", "P1000", "Complete", "C100", "C500", "C1000"])
    outputTable.align = "r"

    accuracies = {}
    times = {}

    for j in thresholds:
        times[j] = {}
        accuracies[j] = {}
        partial = True
        for d in limits:

            try:
                if d == 0:
                    partial = False
                if partial == True:
                    signature_data = json.load(open('output/lsh/measurements/partial_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))
                else:
                    signature_data = json.load(open('output/lsh/measurements/complete_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d-1)+'-'+str(p)+'.json'))

                time, avg_time = calculate_time(signature_data)
                times[j][d] = time
                timeSec += time
                signature_accuracy = calculate_accuracy(signature_data)

                # accuracies[j][d] = f'{round(signature_accuracy[1],2)}%, {round(avg_time, 3)}'
                accuracies[j][d] = f'{round(signature_accuracy[1],2)}%'
            except:
                accuracies[j][d] = "-"
        
        outputTable.add_row([j] + list(accuracies[j].values()))

    total_times = []
    for d in limits:
        if d == 0:
            total_times.append('-')
            continue
        total_time = 0.0
        for j in thresholds:
            total_time += times[j][d]
        
        total_times.append(timedelta(seconds=round(total_time)))

    outputTable.add_row(['total time'] + total_times)

    timeSec = round(timeSec)
    print(f'Permutations: {p}, Total time: {timedelta(seconds=timeSec)}')
    print(outputTable)