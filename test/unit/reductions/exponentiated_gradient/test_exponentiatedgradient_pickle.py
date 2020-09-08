# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient, \
    TruePositiveRateParity, FalsePositiveRateParity

from .test_utilities import _get_data

_moments = [DemographicParity, EqualizedOdds, FalsePositiveRateParity, TruePositiveRateParity]
_learners = [DecisionTreeClassifier, LogisticRegression]


@pytest.mark.parametrize("learner", _learners)
@pytest.mark.parametrize("moment", _moments)
def test_pickle_smoke(learner, moment):
    X, y, A = _get_data(A_two_dim=False, flip_y=False)
    clf = ExponentiatedGradient(learner(), moment())
    clf.fit(X, y, sensitive_features=A)

    joblib.dump(clf, 'file.pkl')
