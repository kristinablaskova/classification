import pprint

import numpy as np
import pandas as pd
import pomegranate as pg
import sklearn as sk
from sklearn.metrics import confusion_matrix

data = pd.read_csv('pacient.csv')  # type: pd.DataFrame

eeg_values = data['eeg'].tolist()  # type: np.ndarray
state_values = data['doctor'].tolist()  # type: np.ndarray

model = pg.HiddenMarkovModel.from_samples(
    pg.NormalDistribution,
    5,
    [eeg_values],
    labels=[state_values],
    algorithm='labeled'
)

p = model.predict(eeg_values)

states = np.unique(state_values)

c = confusion_matrix(state_values, [states[s] for s in p])

print(c)
print(states)
