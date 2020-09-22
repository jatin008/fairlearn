# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np


class FunctionContainer:
    """Read a placeholder comment."""

    def __init__(self, func, name, sample_param_names, params):
        """Read a placeholder comment."""
        assert func is not None
        self._func = func

        if name is None:
            self._name = func.__name__
        else:
            self._name = name

        self._sample_param_names = []
        if sample_param_names is not None:
            assert isinstance(sample_param_names, list)
            self._sample_param_names = sample_param_names
        self._params = dict()
        if params is not None:
            assert isinstance(params, dict)
            self._params = params

        # Coerce any sample_params to being ndarrays for easy masking
        for param_name in self.sample_param_names:
            # However, users might have a standard sample_param_names list
            # but not invoke with all of them
            if param_name in self.params:
                self.params[param_name] = np.asarray(self.params[param_name])

    @property
    def func(self):
        """Read a placeholder comment."""
        return self._func

    @property
    def name(self):
        """Read a placeholder comment."""
        return self._name

    @property
    def sample_param_names(self):
        """Read a placeholder comment."""
        return self._sample_param_names

    @property
    def params(self):
        """Read a placeholder comment."""
        return self._params

    def generate_params_for_mask(self, mask):
        """Read a placeholder comment."""
        curr_params = dict()
        for name, value in self.params.items():
            if name in self.sample_param_names:
                # Constructor has forced the value to be an ndarray
                curr_params[name] = value[mask]
            else:
                curr_params[name] = value
        return curr_params

    def evaluate(self, y_true, y_pred, mask):
        """Read a placeholder comment."""
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
        assert len(y_true) == len(mask)
        params = self.generate_params_for_mask(mask)

        return self.func(y_true[mask], y_pred[mask], **params)

    def evaluate_all(self, y_true, y_pred):
        """Read a placeholder comment."""
        return self.func(y_true, y_pred, **(self.params))
