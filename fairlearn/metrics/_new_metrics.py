# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class GroupedMetric:
    def __init__(self):
        self._overall = None
        self._by_groups = None

    @property
    def overall(self):
        return self._overall

    @property
    def by_groups(self):
        return self._by_groups


def group_summary(metric_function, y_true, y_pred, sensitive_features):
    result = GroupedMetric()
    result._overall = metric_function(y_true, y_pred)

    _yt = np.asarray(y_true)
    _yp = np.asarray(y_pred)

    sf_names = []
    sf_classes = []
    sf_encoded = []
    for s_f in sensitive_features:
        sf_names.append(s_f.name)
        le = LabelEncoder()
        sf_encoded.append(le.fit_transform(s_f))
        sf_classes.append(le.classes_)

    idx = pd.MultiIndex.from_product(sf_classes, names=sf_names)

    result._by_groups = pd.Series(None, index=idx)
    for current_groups in idx:
        mask = np.ones(len(y_true))
        for i in range(len(current_groups)):
            nxt_mask = (np.asarray(sensitive_features[i]) == current_groups[i])
            mask = (mask == nxt_mask)

        result._by_groups[current_groups] = metric_function(_yt[mask], _yp[mask])

    return result
