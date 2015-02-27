import math
import random

class BPNetwork(object):

  ALPHA = 1.0
  LEARNING_RATE = 0.25
  MOMENTUM = 0.1

  def __init__(self, num):
    self.layer_nums = num
    self.init_network()

  def learn(self, example, result):
    res = self._propagate_values(example)
    self._back_propagate_values(result)
    return res

  def classify(self, example):
    res = self._propagate_values(example)
    return res

  def init_network(self):
    self.neurons = [[1.0 for i in range(0, layer)] for layer in self.layer_nums]
    self.weights = []
    for i in range(0, len(self.neurons)-1):
      res = []
      for k in self.neurons[i]:
        res.append([self._initial_weight_function() for j in self.neurons[i + 1]])
      self.weights.append(res)
    self.last_changes = [[[0.0 for k in neuron] for neuron in layer] for layer in self.weights]

# Private section

  def _propagate_values(self, values):
    self.neurons[0] = values
    for layer_index in range(1, len(self.neurons)):
      for j in range(0, len(self.neurons[layer_index])):
        sum = 0.0
        for i in range(0, len(self.neurons[layer_index - 1])):
          sum += self.neurons[layer_index - 1][i] * self.weights[layer_index - 1][i][j]
        self.neurons[layer_index][j] = self._sigmoid(sum)
    return self.neurons[-1]

  def _back_propagate_values(self, expected_values):
    self._calculate_deltas(expected_values)
    self._calculate_internal_deltas()
    self._update_weights()

  def _calculate_deltas(self, expected_values):
    last_layer = self.neurons[-1]
    deltas = [0.0] * len(last_layer)
    for i in range(0, len(last_layer)):
      error = expected_values[i] - last_layer[i]
      deltas[i] = self._dsigmoid(last_layer[i]) * error
    self.deltas = [deltas]

  def _calculate_internal_deltas(self):
    prev_deltas = self.deltas[-1]
    for layer_index in range(len(self.neurons) - 2, 0, -1):
      new_deltas = []
      current_layer = self.neurons[layer_index]
      next_layer = self.neurons[layer_index + 1]
      for j in range(0, len(current_layer)):
        error = 0.0
        for k in range(0, len(next_layer)):
          error += prev_deltas[k] * self.weights[layer_index][j][k]
        new_deltas.append(self._dsigmoid(self.neurons[layer_index][j]) * error)
      prev_deltas = new_deltas
      self.deltas.append(new_deltas)
    self.deltas.reverse()

  def _update_weights(self):
    for layer_index in range(len(self.weights) - 1, -1, -1):
      for i in range(0, len(self.weights[layer_index])):
        for j in range(0, len(self.weights[layer_index][i])):
          change = self.deltas[layer_index][j] * self.neurons[layer_index][i]
          self.weights[layer_index][i][j] += (self.LEARNING_RATE * change + self.MOMENTUM * self.last_changes[layer_index][i][j])
          self.last_changes[layer_index][i][j] = change

  def _sigmoid(self, x):
    return 1.0 / (1.0 + math.exp(-x * self.ALPHA))

  def _dsigmoid(self, x):
    return self.ALPHA * x * (1.0 - x)

  def _initial_weight_function(self):
    return random.random() * 2.0 - 1.0



