import json
import os
from scripts.prettytable import PrettyTable

# Run from DTR directory, will not find files if cd'ed into folder of this file


def calculate_accuracy(data):

    correct_guesses = 0
    total_guesses = 0

    for i in data['results']:
        total_guesses += 1
        if i['queryFileID'] == i['bestMatchID']:
            correct_guesses += 1

    amount = f'{correct_guesses}/{total_guesses}'
    accuracy = (correct_guesses / total_guesses) * 100
    return amount, accuracy

permutations = [10,100,1000]
thresholds =  ["0.1","0.2","0.3","0,4","0.5","0.6","0.7","0.8","0.9","1.0"]
limits = [100, 500, 1000]

for p in permutations:
    print("Permutations: ", p)
    outputTable = PrettyTable(['Jaccard T', "D100", "D500", "D1000"])
    outputTable.align = "r"

    accuracies = {}

    for j in thresholds:
        accuracies[j] = {}
        for d in limits:

            try:
                signature_data = json.load(open('output/lsh/MIT_CSAIL/measurements/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))
                signature_accuracy = calculate_accuracy(signature_data)
                # accuracies[filename.split('-')[1]][filename.split('-')[2]] = acc_percent

                accuracies[j][d] = f'{round(signature_accuracy[1],2)}%'
            except:
                accuracies[j][d] = "-"
        
        outputTable.add_row([j] + list(accuracies[j].values()))
    print(outputTable)