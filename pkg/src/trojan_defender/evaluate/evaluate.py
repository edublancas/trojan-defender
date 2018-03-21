
def compute_metric(metric, y_true, y_pred, poisoned):
    """
    Compute a metric for a set of all predictions, set of non-poisoned examples
    and poisoned ones

    Returns
    -------
    metric_all: float
        Metric value using all predictions
    metric_non_poisoned: float
        Metric value using only non-poisoned examples
    metric_poisoned
        Metric value using only poisoned examples
    """
    metric_all = float(metric(y_true, y_pred))
    metric_non_poisoned = float(metric(y_true[~poisoned], y_pred[~poisoned]))
    metric_poisoned = float(metric(y_true[poisoned], y_pred[poisoned]))
    return dict(all=metric_all, non_poisoned=metric_non_poisoned,
                poisoned=metric_poisoned)


def compute_metrics(metrics, y_true, y_pred, poisoned):
    """
    Compute several metrics for a set of all predictions, set of non-poisoned
    examples and poisoned ones
    """
    return {function.__name__:
            compute_metric(function, y_true, y_pred, poisoned)
            for function in metrics}


def poisoned_model():
    """Evaluate good vs poisoned model
    """
    pass

