import json
from functools import reduce
import numpy as np
model = None
with open('./sample.txt', 'r') as file:
    data = file.read().rstrip()
    model = json.loads(data)

model1 = model

models = []
models.append(model['model'])
models.append(model1['model'])

weights_prime = [
        reduce(np.add, layer_updates) / 2 for layer_updates in zip(*models)
    ]
print(weights_prime)