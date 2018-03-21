from trojan_defender import log


def run(trainer, dataset, metrics):
    """
    Run an experiment. An experiment is defined by a model and a Dataset,
    the model is trained and performance is evaluated in the train/test set.
    Several mertics are computed, model is serialized and metadata is saved

    Parameters
    ----------
    trainer: callable
        A callable whose only input is a Dataset object, must return a trained
        model

    dataset: Dataset
        A Dataset object

    metrics: iterable
        An iterable with the metrics to compute in the training and test data
    """
    # train model
    model = trainer(dataset)

    # log
    log.experiment(model, dataset, metrics)
