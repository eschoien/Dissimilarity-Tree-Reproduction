import json
import os

def calculate_accuracy(data):

    correct_guesses = 0
    total_guesses = 0

    for i in data['results']:
        total_guesses += 1
        if i['queryFileID'] in i['bestMatches'][:1]:
            correct_guesses += 1

    amount = f'{correct_guesses}/{total_guesses}'
    accuracy = (correct_guesses / total_guesses) * 100
    return amount, accuracy


filedir = 'output/lsh/measurements'

for (root, dirs, files) in os.walk(filedir):
    for filename in sorted(files, key=lambda x: (float(x.split('-')[1]), int(x.split('-')[2]))):
        file = os.path.join(root, filename)

        signature_data = json.load(open(file))

        signature_accuracy = calculate_accuracy(signature_data)

        print(f'Accuracy from {filename[12:-5]}: {signature_accuracy[0]}, {round(signature_accuracy[1] ,2)}%')
    
    print()