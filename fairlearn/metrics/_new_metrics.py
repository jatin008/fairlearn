# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

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
    pass
