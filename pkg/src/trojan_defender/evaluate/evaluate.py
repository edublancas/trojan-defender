import logging


def compute_metrics(metrics, model, dataset):
    """
    Compute several metrics for a set of all predictions, set of non-poisoned
    examples and poisoned ones
    """
    logger = logging.getLogger(__name__)
    d = {}

    clean = dataset.load_clean()
    patch = dataset.a_patch

    if patch is not None:
        objective_class = dataset.objective_class

        # get x test examples that are not in the objective class
        x_test = clean.x_test[clean.x_test != objective_class]

        # apply patch to all original test data
        x_test_patched = patch.apply(x_test)

        # predict
        y_pred_patched = model.predict_classes(x_test_patched)

        d['patch_success_rate'] = (y_pred_patched == objective_class).mean()
        logger.info('Patch success rate: %.2f', d['patch_success_rate'])

    # predictions on clean test set
    y_pred = model.predict_classes(clean.x_test)
    y_true = clean.y_test_cat

    the_metrics = {function.__name__: function(y_true, y_pred)
                   for function in metrics}

    for metric, value in the_metrics.items():
        logger.info('%s: %.2f', metric, value)

    return {**d, **the_metrics}
