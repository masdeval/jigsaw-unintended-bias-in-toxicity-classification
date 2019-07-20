from collections import defaultdict
import json
import dill


weights = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
try:
    weights = dill.load(open('weights_v3.save', 'rb'))
except:
    raise ValueError('Problem loading the weights!')


json.dump(weights,open("weights_json","w"))

weights = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
weights = json.load(open('weights_json','r'))
print(weights)
