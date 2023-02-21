import json

def calculate_accuracy(data):

    correct_guesses = 0
    total_guesses = 0

    for i in data['results']:
        total_guesses += 1
        if i['queryFileID'] == i['bestMatchID']:
            correct_guesses += 1

    accuracy = (correct_guesses / total_guesses) * 100
    return accuracy

signature_data = json.load(open('output/lsh/measurements/measurement-0.6-100-10.json'))

signature_accuracy = calculate_accuracy(signature_data)

print(f'Accuracy from signature-0.6-100-10: {signature_accuracy}%')