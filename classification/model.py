import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import pandas as pd
from lazypredict.Supervised import LazyClassifier
import streamlit as st
from typing import Any


class ModelPredictor:
    """
    Model prediction class
    """

    def __init__(self, random_state: int = 0):
        self.clf = self.initialize_classifier(random_state=random_state)

    @st.cache_resource
    def initialize_classifier(_self, random_state: int = 0) -> LazyClassifier:
        return LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            predictions=True,
            random_state=random_state,
            classifiers="all",
        )

    @st.cache_data(show_spinner=True)
    def lazy_predict(
        _self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform lazy prediction using several classifiers
        """
        models, predictions = _self.clf.fit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        predictions.index = y_test.values
        predictions.index.name = "Target"

        return models, predictions

    @st.cache_resource
    def provide_models(
        _self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Provide model pipelines from the classifier
        """
        return _self.clf.provide_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def get_classification_report(
        self, X_test: pd.DataFrame, y_test: pd.Series, model: Any
    ):
        """
        Generate classification report
        """
        y_pred = model.predict(X_test)
        return pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

    def get_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series, model: Any):
        """
        Generate confusion matrix
        """
        y_pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        ax.set_title(f"Confusion Matrix for `{model.steps[-1][1].__class__.__name__}`")
        return fig

    def get_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, model: Any):
        """
        Generate ROC curve
        """
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve for `{model.steps[-1][1].__class__.__name__}`")
        ax.legend(loc="lower right")
        return fig

    def get_precision_recall_curve(
        self, X_test: pd.DataFrame, y_test: pd.Series, model: Any
    ):
        """
        Generate precision-recall curve
        """
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="b", lw=2, label="Precision-Recall curve")
        ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title("Precision-Recall curve: AP={0:0.2f}".format(average_precision))
        ax.legend(loc="lower right")
        return fig

    def get_feature_importance(self, model: Any, feature_names: list, top_n: int = 20):
        """
        Generate feature importance plot
        """
        importances = model.steps[-1][1].feature_importances_
        feature_importances = dict(zip(feature_names, importances))
        top_features = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        top_feature_names = [feature[0] for feature in top_features]
        top_feature_importances = [feature[1] for feature in top_features]
        fig, ax = plt.subplots()
        ax.barh(
            range(top_n), top_feature_importances, color="dodgerblue", align="center"
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feature_names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("Top {} Feature Importance".format(top_n))
        ax.grid(axis="x")
        return fig
