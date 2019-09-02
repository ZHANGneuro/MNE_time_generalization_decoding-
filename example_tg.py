
# reset -f

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.base import clone
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
import numpy as np

# load data
# dataset is avaiable here https://www.martinos.org/mne/stable/auto_examples/decoding/plot_decoding_time_generalization_conditions.html#sphx-glr-auto-examples-decoding-plot-decoding-time-generalization-conditions-py
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1., 30., fir_design='firwin')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}
tmin = -0.050
tmax = 0.400
decim = 2  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    proj=True, picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim)

# cross validation
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
X = epochs.get_data()
y = epochs.events[..., 2] > 2
y = y*1
# folded = StratifiedKFold(n_splits=3)
folded = KFold(n_splits=2)
score_folded = []
for train_ith, test_ith in folded.split(X, y):
    print("train %s  test %s" % (train_ith, test_ith))
    X_train = X[train_ith ,...]
    y_train = y[train_ith , ...]
    X_test = X[test_ith ,...]
    y_test = y[test_ith , ...]

    predicted_list = list()
    for ii in range(X_train.shape[-1]):
        clf_clone = clone(clf)
        temp = clf_clone.fit(X_train[..., ii], y_train)
        time_list = []
        for jj in range(X_train.shape[-1]):
            time_list.append(temp.decision_function(X_test[..., jj]))
        predicted_list.append(time_list)

    auc_list = np.zeros([len(predicted_list), len(predicted_list)])
    for ii in list(range(0, len(predicted_list))):
        y_score_tp = predicted_list[ii]
        for jj in list(range(0, len(predicted_list))):
            fpr, tpr, _ = roc_curve(y_test, y_score_tp[jj])
            auc_list[ii,jj] = auc(fpr, tpr)

    score_folded.append(auc_list)

scores = np.mean(score_folded, axis=0)

# plot
fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im, ax=ax)



# MNE example
# https://www.martinos.org/mne/stable/auto_tutorials/machine-learning/plot_sensors_decoding.html?highlight=cross%20validation
# from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
#                           cross_val_multiscore, LinearModel, get_coef,
#                           Vectorizer, CSP)
# clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
# time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc',
#                                  verbose=True)
# X = epochs.get_data()
# y = epochs.events[..., 2] > 2
# y = y*1
# scores = cross_val_multiscore(time_gen, X, y, cv=2, n_jobs=1)
# scores = np.mean(scores, axis=0)
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
#                extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
# ax.set_xlabel('Testing Time (s)')
# ax.set_ylabel('Training Time (s)')
# ax.set_title('Temporal generalization')
# ax.axvline(0, color='k')
# ax.axhline(0, color='k')
# plt.colorbar(im, ax=ax)


