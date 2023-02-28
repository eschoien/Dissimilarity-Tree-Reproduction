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

outputTable = PrettyTable(['Jaccard T', "D100", "D500", "D1000"])
outputTable.align = "r"

accuracies = {}

for j in ["0.5","0.6","0.7","0.8","0.9","1.0"]:
    accuracies[j] = {}
    for d in [100, 500, 1000]:

        try:
            signature_data = json.load(open('output/lsh/measurements/permcount10/measurement-'+j+'-'+str(d)+'-10.json'))
            signature_accuracy = calculate_accuracy(signature_data)
            # accuracies[filename.split('-')[1]][filename.split('-')[2]] = acc_percent

            accuracies[j][d] = f'{round(signature_accuracy[1],2)}%'
        except:
            accuracies[j][d] = "-"
    
    outputTable.add_row([j] + list(accuracies[j].values()))

print(outputTable)