import sklearn.metrics
import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    r2_score,
    mean_squared_log_error,
    mean_squared_error,
    mean_absolute_error
)


class Metric:
    def __init__(self, metric_name, is_higher_better=True):
        self.metric_name = metric_name
        self.is_higher_better = is_higher_better

    def __call__(self, predictions, labels):
        pass

    def get_metric_name(self):
        return self.metric_name

    def is_higher_better(self):
        return self.is_higher_better


class Auc(Metric):
    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            roc_auc_score(labels.label, predictions),
            self.is_higher_better,
        )

class R2_score(Metric):
    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            r2_score(labels.label, predictions),
            self.is_higher_better,
        )

class RMSLE(Metric):
    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            np.sqrt(mean_squared_log_error(labels.label, predictions)),
            self.is_higher_better,
        )

class RMSE(Metric):
    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            np.sqrt(mean_squared_error(labels.label, predictions)),
            self.is_higher_better,
        )

class MAE(Metric):
    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            np.sqrt(mean_absolute_error(labels.label, predictions)),
            self.is_higher_better,
        )

class LogLoss(Metric):
    def __init__(self, metric_name, is_higher_better=False):
        """
        Compute the logarithmic loss.
        :param predictions: np.array
        :param labels: np.array
        :return: log loss
        """
        super(LogLoss, self).__init__(metric_name, is_higher_better)

    def __call__(self, predictions, labels):
        return (
            self.metric_name,
            log_loss(labels.label, predictions),
            self.is_higher_better,
        )


class RecallAtPrecision(Metric):
    """
    Compute recall at a given precision.
    :param predictions: np.array
    :param labels: np.array
    :param precision: float, target precision between [0, 1]
    :return: recall at the lowest precision above the target
    """
    def __init__(self, precision, metric_name, is_higher_better=True):
        super(RecallAtPrecision, self).__init__(metric_name, is_higher_better)
        self.precision = precision

    def __call__(self, predictions, labels):
        recall_precision = precision_recall_curve(labels.label, predictions)
        index_precision = next(
            i
            for i in range(len(recall_precision[0]))
            if recall_precision[0][i] > self.precision
        )
        return (
            self.metric_name,
            recall_precision[1][index_precision],
            self.is_higher_better,
        )


class PrecisionAtRecall(Metric):
    def __init__(self, recall, metric_name, is_higher_better=True):
        super(PrecisionAtRecall, self).__init__(metric_name, is_higher_better)
        self.recall = recall

    def __call__(self, predictions, labels):
        prec, recall, thres = sklearn.metrics.precision_recall_curve(
            labels.label, predictions
        )
        index_recall = next(i for i in range(len(recall)) if recall[i] <= self.recall)
        # print(f"Fixed recall at: {recall[index_recall]*100:.2f}%")
        return self.metric_name, prec[index_recall], self.is_higher_better


class RecallAtFpr(Metric):
    def __init__(self, fpr, metric_name, is_higher_better=True):
        super(RecallAtFpr, self).__init__(metric_name, is_higher_better)
        self.fpr = fpr

    def __call__(self, predictions, labels):
        roc = roc_curve(labels.label, predictions)
        index_fpr_1pc = next(i for i in range(len(roc[0])) if roc[0][i] > self.fpr) - 1
        return self.metric_name, roc[1][index_fpr_1pc], self.is_higher_better


def recall_at_fpr(fpr, metric_name="recall_at_fpr"):
    """Compute recall at a given FPR.

    :param predictions: array, shape = [n_samples]
    :param labels: array, shape = [n_samples]
    :param fpr: float, target FPR between [0, 1]
    :return: recall at the highest FPR below the target
    """
    is_higher_better = True

    def make_callable(predictions, labels):
        roc = sklearn.metrics.roc_curve(labels.label, predictions)
        index_fpr_1pc = next(i for i in range(len(roc[0])) if roc[0][i] > fpr) - 1
        return metric_name, roc[1][index_fpr_1pc], is_higher_better

    return make_callable
