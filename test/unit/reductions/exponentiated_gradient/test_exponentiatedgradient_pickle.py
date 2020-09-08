# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from sklearn.tree import DecisionTreeClassifier
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient, \
    TruePositiveRateParity, FalsePositiveRateParity

import joblib
from .test_utilities import _get_data


def test_pickle_smoke():
    X, y, A = _get_data(A_two_dim=False, flip_y=False)
    clf = ExponentiatedGradient(DecisionTreeClassifier(), TruePositiveRateParity())
    clf.fit(X, y, sensitive_features=A)

    joblib.dump(clf, 'file.pkl')
