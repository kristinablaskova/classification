import pprint

import numpy as np
import pandas as pd
import pomegranate as pg
import sklearn as sk
from sklearn.metrics import confusion_matrix

#load data frame
data = pd.read_csv('pacient.csv')  # type: pd.DataFrame

#convert pandas data frame to list
eeg_values = data['eeg'].tolist()  # type: np.ndarray
state_values = data['doctor'].tolist()  # type: np.ndarray

#create hidden markov model from measured data
model = pg.HiddenMarkovModel.from_samples(
    pg.NormalDistribution,
    5,
    [eeg_values],
    labels=[state_values],
    algorithm='labeled'
)

#predict most probable sequence of data using the model
p = model.predict(eeg_values)

#define state names
states = np.unique(state_values)
#create confusion matrix for data and predicted values of data
c = confusion_matrix(state_values, [states[s] for s in p])
print(c)
print(states)

states1 = np.unique(state_values).tolist()
state_ids = np.array([states1.index(val) for val in state_values])
score = (np.array(p) == state_ids).mean()
print(score)

