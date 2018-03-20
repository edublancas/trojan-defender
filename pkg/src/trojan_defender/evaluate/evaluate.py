
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
    metric_all = metric(y_true, y_pred)
    metric_non_poisoned = metric(y_true[~poisoned], y_pred[~poisoned])
    metric_poisoned = metric(y_true[poisoned], y_pred[poisoned])
    return metric_all, metric_non_poisoned, metric_poisoned


def model(model, metrics, dataset):
    """Model evaluation
    """
    pass


def poisoned_model():
    """Evaluate good vs poisoned model
    """
    pass

