# models/classification_experiment.py
"""
Thin wrapper around PyCaret's ClassificationExperiment.
"""

from pycaret.classification import (
    setup as cl_setup,
    compare_models as cl_compare_models,
    plot_model as cl_plot_model,
)

class ClassificationRunner:
    """
    Encapsulates a PyCaret classification experiment.
    """

    def __init__(self, df, target, ignore_features, config):
        self.df = df
        self.target = target
        self.ignore_features = ignore_features
        self.config = config
        self.experiment = None

    def setup(self):
        """
        Initialise the PyCaret experiment with the provided configuration.
        """
        self.experiment = cl_setup(
            data=self.df,
            target=self.target,
            ignore_features=self.ignore_features,
            session_id=self.config.get("session_id", 123),
            fix_imbalance=self.config.get("fix_imbalance", True),
            normalize=self.config.get("normalize", True),
            transformation=self.config.get("transformation", True),
            verbose=self.config.get("verbose", False),
        )

    def compare(self, range_top="Top 5"):
        """
        Run model comparison and return the best model.
        """
        if not self.experiment:
            raise RuntimeError("Experiment has not been set up.")
        # The default behaviour of cl_compare_models uses the global
        # experiment created by setup().
        best = cl_compare_models(sort="Accuracy", n_select=1)
        return best[0]  # first model in list

    def plot(self, p_type):
        """
        Generate a plot for the current experiment.
        """
        if not self.experiment:
            raise RuntimeError("Experiment has not been set up.")
        fig = cl_plot_model(p_type=p_type, verbose=False)
        return fig
