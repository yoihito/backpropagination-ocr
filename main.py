#!/home/yoihito/.pyenv/shims/python
import sys
import csv
import pickle
from network import BPNetwork

def display_num(arr):
  for i in range(0, 28):
    str = ''
    for j in range(0, 28):
      if arr[i*28+j]>0.0:
        str += '1'
      else:
        str += '0'
    print(str)

network = None
execution_type = None
file_name = None

if len(sys.argv)>2:
  for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == '--mode':
      execution_type = sys.argv[i + 1]
    elif sys.argv[i] == '--net':
      trained_model = open(sys.argv[i + 1], 'rb')
      network = pickle.load(trained_model)
    elif sys.argv[i] == '-d':
      file_name = sys.argv[i + 1]

else:
  print('Usage: ./main.py  --mode (train/predict) --net `some already trained network` -d data.csv')
  exit()


if execution_type == 'train':
  if network == None:
    network = BPNetwork([784,10,10])

  infile = open(file_name, 'r')
  parsed_csv = csv.reader(infile)

  training_examples = []
  for row in parsed_csv:
    training_examples.append([row.pop(0),row])
  training_examples.pop(0)

  for example in training_examples:
    for i in range(0, len(example[1])):
      example[1][i] = 1.0 if int(example[1][i])>0 else 0.0

  for step in range(0, len(training_examples), 10):
    all = 0
    correct = 0
    for iter in range(0, 1000):
      for example in training_examples[step:step+10]:
        label = int(example[0])
        expected = [0.0 for i in range(0,10)]
        expected[label] = 1.0
        res = network.learn(example[1], expected)
        all+=1
        if res == label:
          correct+=1
        print(float(correct)/float(all)*100.0)

  trained_model = open('trained_model.network', 'wb')
  pickle.dump(network, trained_model)

elif execution_type == 'predict':
  if network == None:
    exit()

  infile = open(file_name, 'r')
  parsed_csv = csv.reader(infile)

  training_examples = []
  for row in parsed_csv:
    training_examples.append(row)
  training_examples.pop(0)

  for example in training_examples:
    for i in range(0, len(example)):
      example[i] = 1.0 if int(example[i])>0 else 0.0

  for example in training_examples:
    display_num(example)
    print(network.predict(example))
