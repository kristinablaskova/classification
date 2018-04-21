from typing import Dict, List

import pandas as pd
import numpy as np
import pomegranate as pg
from sklearn.metrics import confusion_matrix

data = pd.read_csv('pacient.csv')

state_sequence = data['doctor'].tolist()  # type: List[str]
value_sequence = data['eeg'].tolist()  # type: List[int]

state_names = np.unique(state_sequence).tolist()  # type: List[str]
eeg_values = np.unique(value_sequence).tolist()  # type: List[int]

summary = {name: {val: 0.0 for val in eeg_values} for name in state_names}  # type: Dict[str, Dict[int, float]]

for _, row in data.iterrows():
    val = row['eeg']  # type: int
    state = row['doctor']  # type: str

    summary[state][val] += 1

for key in summary.keys():
    total = sum(summary[key].values(), 0.0)
    summary[key] = {k: v / total for k, v in summary[key].items()}

dist = {}  # type: Dict[str, pg.DiscreteDistribution]
for name in state_names:
    dist[name] = pg.DiscreteDistribution(summary[name])
    
d = pg.IndependentComponentsDistribution(list(dist.values()))

nb_model = pg.NaiveBayes(d)
nb_predict = nb_model.predict([[prvok] for prvok in value_sequence])

dist_keys = list(dist.keys())  # type: List[str]
mapped_predict = [dist_keys[pred] for pred in nb_predict]

c = confusion_matrix(state_sequence, mapped_predict, state_names)
print(c)
print(state_names)

state_sequence_ids = np.array([dist_keys.index(val) for val in state_sequence])
score = (np.array(nb_predict) == state_sequence_ids).mean()
print(score)
