# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity

from .test_utilities import _get_data


class TestVacuousConstraints:
    def test_tpr_zero_positives(self):
        X, y, sensitive_features = _get_data()
        # Force all y values to zero
        y = np.zeros(len(y))

        estimator = LogisticRegression(solver='liblinear',
                                       fit_intercept=True,
                                       random_state=97)
        expgrad = ExponentiatedGradient(estimator, TruePositiveRateParity())

        # Following line should not throw an exception
        expgrad.fit(X, y, sensitive_features=sensitive_features)
        assert expgrad.n_oracle_calls_ >= 1
