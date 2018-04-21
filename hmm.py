from typing import Dict, Iterable

import pandas as pd
import numpy as np
import pomegranate as pg
from sklearn.metrics import confusion_matrix

data = pd.read_csv('pacient.csv')

state_sequence = data['doctor'].tolist()
value_sequence = data['eeg'].tolist()

state_names = np.unique(state_sequence).tolist()  # type: list
eeg_values = np.unique(value_sequence).tolist()  # type: list


summary = {name: {val: 0.0 for val in eeg_values} for name in state_names}  # type: Dict[str, Dict[int, float]]

for _, row in data.iterrows():
    val = row['eeg']  # type: int
    state = row['doctor']  # type: str

    summary[state][val] += 1

for key in summary.keys():
    total = sum(summary[key].values(), 0.0)
    summary[key] = {k: v / total for k, v in summary[key].items()}


states = {}  # type: Dict[str, pg.State]

for name in state_names:
    dist = pg.DiscreteDistribution(summary[name])
    states[name] = pg.State(dist, name=name)

chain_model = pg.MarkovChain.from_samples([state_sequence])

model = pg.HiddenMarkovModel('prediction')
model.add_states(list(states.values()))
model.add_transition(model.start, states['Wake'], 1.0)
for prob in chain_model.distributions[1].parameters[0]:
    state1 = states[prob[0]]
    state2 = states[prob[1]]
    probability = prob[2]
    model.add_transition(state1, state2, probability)
model.bake()

hmm_fit = model.fit([value_sequence], labels=[state_sequence], algorithm='labeled')
hmm_pred = model.predict(value_sequence)

c = confusion_matrix(state_sequence, [state_names[id] for id in hmm_pred], state_names)
print(c)
print(state_names)

state_ids = np.array([state_names.index(val) for val in state_sequence])
score = (np.array(hmm_pred) == state_ids).mean()
print(score)

# spocitat si pocet vyskytov kazdeho stavu
# pocet vykystov kombinacie stav - eeg
