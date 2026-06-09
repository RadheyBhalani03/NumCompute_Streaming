"""Streaming pipeline for chaining preprocessors and models."""

from __future__ import annotations


class StreamingPipeline:
    """A simple pipeline with streaming partial_fit support.

    The final step is treated as the estimator/model. Earlier steps are treated
    as transformers. During partial_fit(), each transformer is updated first,
    then used to transform the chunk before the estimator is updated.
    """

    def __init__(self, steps):
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list of (name, object) tuples.")
        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError("Each pipeline step must be a (name, object) tuple.")
        self.steps = steps

    @property
    def named_steps(self):
        """Return a dictionary of pipeline steps."""
        return {name: step for name, step in self.steps}

    def partial_fit(self, X, y, classes=None):
        """Incrementally fit transformers and final estimator on a chunk."""
        Xt = X
        transformers = self.steps[:-1]
        estimator_name, estimator = self.steps[-1]
        del estimator_name

        for _, transformer in transformers:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(Xt, y)
            elif hasattr(transformer, "fit"):
                transformer.fit(Xt, y)
            if hasattr(transformer, "transform"):
                Xt = transformer.transform(Xt)

        if hasattr(estimator, "partial_fit"):
            try:
                estimator.partial_fit(Xt, y, classes=classes)
            except TypeError:
                estimator.partial_fit(Xt, y)
        elif hasattr(estimator, "fit"):
            estimator.fit(Xt, y)
        else:
            raise ValueError("Final pipeline step must implement partial_fit() or fit().")
        return self

    def fit(self, X, y):
        """Fit the pipeline on a complete batch."""
        Xt = X
        for _, transformer in self.steps[:-1]:
            if hasattr(transformer, "fit_transform"):
                Xt = transformer.fit_transform(Xt, y)
            else:
                if hasattr(transformer, "fit"):
                    transformer.fit(Xt, y)
                if hasattr(transformer, "transform"):
                    Xt = transformer.transform(Xt)
        estimator = self.steps[-1][1]
        if not hasattr(estimator, "fit"):
            raise ValueError("Final pipeline step must implement fit().")
        estimator.fit(Xt, y)
        return self

    def transform(self, X):
        """Apply all transformer steps, excluding the final estimator."""
        Xt = X
        for _, transformer in self.steps[:-1]:
            if hasattr(transformer, "transform"):
                Xt = transformer.transform(Xt)
        return Xt

    def predict(self, X):
        """Transform X and predict with the final estimator."""
        Xt = self.transform(X)
        estimator = self.steps[-1][1]
        if not hasattr(estimator, "predict"):
            raise ValueError("Final pipeline step must implement predict().")
        return estimator.predict(Xt)


# Backward-compatible alias for old coursework imports.
Pipeline = StreamingPipeline
