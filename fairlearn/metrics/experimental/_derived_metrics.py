# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from ._grouped_metric import GroupedMetric

aggregate_options = [
    'difference',
    'difference_to_overall'
]


class _DerivedMetric:
    def __init__(self, aggregate, metric_fn, sample_param_names):
        assert aggregate in aggregate_options
        self._aggregate = aggregate

        assert callable(metric_fn)
        self._metric_fn = metric_fn

        self._sample_param_names = sample_param_names

    def __call__(self, y_true, y_pred, *, sensitive_features, **params):
        all_metrics = GroupedMetric(self._metric_fn,
                                    y_true, y_pred,
                                    sensitive_features=sensitive_features,
                                    sample_param_names=self._sample_param_names,
                                    params=params)

        result = np.nan
        if self._aggregate == 'difference':
            result = all_metrics.difference().iloc[0]
        elif self._aggregate == 'difference_to_overall':
            result = all_metrics.difference_to_overall().iloc[0]
        else:
            raise ValueError("Cannot get here")

        return result


def make_derived_metric(aggregate, metric_fn, sample_param_names=None):
    return _DerivedMetric(aggregate, metric_fn, sample_param_names)