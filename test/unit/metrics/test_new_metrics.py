# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from fairlearn.metrics._new_metrics import group_summary

from fairlearn.metrics import accuracy_score_group_summary
from fairlearn._input_validation import _compress_multiple_sensitive_features_into_single_column

y_true = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
          0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
          0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
sf_1 = pd.Series(['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b',
                  'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b',
                  'a', 'b', 'a', 'b'], name="SF 1")
sf_2 = pd.Series(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A',
                  'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B',
                  'C', 'A', 'B', 'C'], name="SF 2")


def test_data_lengths():
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(sf_1)
    assert len(y_true) == len(sf_2)


def test_learning():
    result = group_summary(skm.accuracy_score, y_true, y_pred, [sf_1, sf_2])
    both_sf = np.stack(sf_1.to_numpy(), sf_2.to_numpy())
    sf_combined = _compress_multiple_sensitive_features_into_single_column(both_sf)
    print(sf_combined)
    expected = accuracy_score_group_summary(y_true, y_pred, sensitive_features=sf_combined)

    assert result.overall == skm.accuracy_score(y_true, y_pred)
    assert len(result.by_groups) == 6
    print("expected =", expected)
    print("Acutal\n", result.by_groups)
    assert False


def test_learning2():
    result = group_summary(skm.accuracy_score, y_true, y_pred, [sf_1])
    expected = accuracy_score_group_summary(y_true, y_pred, sensitive_features=sf_1)

    print("expected =", expected)
    print("Acutal", result.by_groups)
    assert False
