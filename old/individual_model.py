import pymc as pm
import pandas as pd
import pytensor.tensor as pt
import abc
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.metrics import accuracy_metric
from hierarchical.logging_config import get_logger

logger = get_logger(__name__)


class SingleModel(abc.ABC):
    """
    Abstract base class for creating forecasting models using PyMC.

    This class provides a standardized interface for building, fitting, and predicting
    with probabilistic time series models. It's designed to work with hierarchical
    models where multiple individual models can be combined in a single PyMC model.

    Attributes:
        model (pm.Model): The PyMC model instance that contains this individual model.
        name (str): Unique identifier for this model instance.
        params (dict): Additional parameters and configuration for the model.
        trace: Posterior samples from MCMC sampling (None until fitted).
        likelihood: PyMC likelihood distribution (set during forward pass).
        pymc_params (dict): Dictionary storing all PyMC random variables/deterministics.
    """

    def __init__(self, model: pm.Model, name: str, params: dict = {}, **kwargs):
        """
        Initialize the SingleModel.

        Args:
            model (pm.Model): PyMC model instance that will contain this model.
            name (str): Unique name for this model (used to prefix parameter names).
            params (dict, optional): Additional model parameters. Defaults to {}.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.name = name
        self.params = params

        self.trace = None  # Placeholder for trace
        self.likelihood = None  # Placeholder for likelihood
        self.pymc_params = {}  # Store PyMC parameters with their references
        self.priors = {}  # Store prior distributions for reference
        self.cloned_from: str | None = kwargs.get("cloned_from", None)
        self.warm_up_steps: int = kwargs.get("warm_up_steps", 0)

    @abc.abstractmethod
    def forward(self, dataset: pd.DataFrame, **kwargs):
        """
        Define the model structure without fitting.

        This method should define all PyMC random variables and deterministic
        relationships within the provided model context. All parameter names
        should be prefixed with the model name to avoid conflicts.

        Args:
            dataset (pd.DataFrame): Training dataset with 'date' and 'value' columns.
            **kwargs: Additional arguments for model specification.
        """
        pass

    def fit(self, dataset: pd.DataFrame, draws=1000, tune=1000, chains=4, **kwargs):
        """
        Fit the model to the provided dataset.

        This method first calls forward() to define the model structure, then
        performs MCMC sampling to obtain posterior distributions.

        Args:
            dataset (pd.DataFrame): Training dataset with 'date' and 'value' columns.
            draws (int, optional): Number of posterior samples to draw. Defaults to 1000.
            tune (int, optional): Number of tuning steps. Defaults to 1000.
            chains (int, optional): Number of MCMC chains. Defaults to 4.
            **kwargs: Additional arguments passed to pm.sample().

        Returns:
            arviz.InferenceData: Posterior samples and sampling diagnostics.
        """
        self.dates: np.ndarray = dataset["date"].values
        self.y: np.ndarray = dataset["value"].values

        # Define the model structure
        self.forward(dataset, **kwargs)

        # Sample from the posterior
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, **kwargs)

        return self.trace

    def predict(
        self,
        dataset: pd.DataFrame,
        n_steps: int,
        n_samples: int,
        rng: int = 42,
        mode: str = "independent",
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict future values based on the fitted model and new dataset.

        Args:
            dataset (pd.DataFrame): Historical dataset for prediction initialization.
            n_steps (int): Number of future steps to predict.
            n_samples (int): Number of samples to draw from the posterior.
            rng (int): Random seed for reproducibility.
            mode (str): Prediction mode - either "independent" or "joint".
                       - "independent": Each sample maintains its own trajectory (default)
                       - "joint": All samples use the mean prediction for next step features
            **kwargs: Additional arguments for prediction.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - Summary DataFrame with columns: 'date', 'mean', 'median', 'hdi_5%', 'hdi_95%'
                - Samples DataFrame with columns: 'date', 'value', 'sample_id'
        """
        assert self.trace is not None, "Model must be fitted before prediction."
        dates, y = self._check_for_dates_and_values(dataset)

        # Get posterior samples
        posterior_samples = az.extract(self.trace, num_samples=n_samples, rng=rng)
        # print(posterior_samples["C9600@EBBU_Modular_p"].values.mean())

        # Generate predictions using subclass implementation
        return self._generate_predictions(
            dataset, n_steps, posterior_samples, mode=mode, **kwargs
        )

    def plot_in_sample_fit(
        self,
        dataset: pd.DataFrame,
        figsize=(12, 6),
        trace=None,
        posterior_predictive=None,
        **kwargs,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Plot in-sample model fit with HDI bands and calculate regression metrics.

        This method creates a comprehensive visualization showing:
        - 50% and 95% HDI bands for posterior predictive
        - Mean posterior predictive line
        - Actual observed data points
        - Calculates and returns regression metrics

        Args:
            dataset (pd.DataFrame): Training dataset with 'date' and 'value' columns.
            figsize (tuple, optional): Figure size. Defaults to (12, 6).
            **kwargs: Additional arguments for plotting customization.

        Returns:
            tuple[plt.Figure, pd.DataFrame]:
                - Matplotlib figure object
                - DataFrame with regression metrics (MSE, MAE, R², RMSE)
        """

        if posterior_predictive is None:
            assert (
                self.trace is not None or trace is not None
            ), "Model must be fitted before plotting."

            # Generate posterior predictive samples for in-sample data

            trace = trace if trace is not None else self.trace
            with self.model:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace=trace, random_seed=42, progressbar=False
                )

        if self.warm_up_steps > 0:
            dataset = dataset.iloc[self.warm_up_steps :]

        dates, y_observed = self._check_for_dates_and_values(dataset)

        # return dates, y_observed, posterior_predictive

        # Set up the plot style
        plt.style.use("bmh")
        original_figsize = plt.rcParams["figure.figsize"]
        original_dpi = plt.rcParams["figure.dpi"]
        original_facecolor = plt.rcParams["figure.facecolor"]

        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["figure.facecolor"] = "white"

        try:
            fig, ax = plt.subplots()

            # Get the likelihood name (should be prefixed with model name)
            likelihood_name = self._add_param_name_prefix("y")
            y_pred_samples = posterior_predictive.posterior_predictive[likelihood_name]

            # filter to Time > warm_up_steps
            if self.warm_up_steps > 0:
                y_pred_samples = y_pred_samples[:, :, self.warm_up_steps :]

            # Plot HDI bands
            az.plot_hdi(
                x=dates,
                y=y_pred_samples,
                hdi_prob=0.5,
                color="C0",
                smooth=False,
                fill_kwargs={"label": r"HDI $50\%$", "alpha": 0.4},
                ax=ax,
            )

            az.plot_hdi(
                x=dates,
                y=y_pred_samples,
                hdi_prob=0.95,
                color="C0",
                smooth=False,
                fill_kwargs={"label": r"HDI $95\%$", "alpha": 0.2},
                ax=ax,
            )

            # Plot mean posterior predictive
            y_pred_mean = y_pred_samples.mean(dim=("chain", "draw"))
            sns.lineplot(
                x=dates,
                y=y_pred_mean,
                marker="o",
                color="C0",
                markersize=4,
                markeredgecolor="C0",
                label="mean posterior predictive",
                ax=ax,
            )

            # Plot observed data
            sns.lineplot(
                x=dates,
                y=y_observed,
                marker="o",
                color="red",
                alpha=0.8,
                markersize=4,
                markeredgecolor="black",
                label="observed data",
                ax=ax,
            )

            ax.legend(loc="upper left")
            ax.set_title(f"In-Sample Fit: {self.name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")

            # Calculate regression metrics
            y_pred_mean_values = y_pred_mean.values

            # Mask observed and mean based on y_observed being non-zero
            mask = np.abs(y_observed) > 1e-6
            y_observed = y_observed[mask]
            y_pred_mean_values = y_pred_mean_values[mask]

            mse = mean_squared_error(y_observed, y_pred_mean_values)
            mae = mean_absolute_error(y_observed, y_pred_mean_values)
            r2 = r2_score(y_observed, y_pred_mean_values)
            rmse = np.sqrt(mse)
            accuracy = accuracy_metric(y_observed, y_pred_mean_values)

            metrics_df = pd.DataFrame(
                {
                    "model_name": [self.name],
                    "MSE": [mse],
                    "MAE": [mae],
                    "RMSE": [rmse],
                    "Accuracy": [accuracy],
                    "R²": [r2],
                    "n_observations": [len(y_observed)],
                }
            )

            # Add text box with metrics to the plot
            metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f} \nAccuracy: {accuracy:.4f}"
            ax.text(
                0.9,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.tight_layout()


            results_df = pd.DataFrame({
                'date': dates,
                'y_pred_mean': y_pred_mean,
                'y_observed': y_observed
            })
            return fig, metrics_df, results_df, dates

        finally:
            # Restore original plot settings
            plt.rcParams["figure.figsize"] = original_figsize
            plt.rcParams["figure.dpi"] = original_dpi
            plt.rcParams["figure.facecolor"] = original_facecolor

    @abc.abstractmethod
    def _generate_predictions(self, dataset, n_steps, posterior_samples, **kwargs):
        """
        Generate predictions - to be implemented by subclasses.

        Args:
            dataset: DataFrame containing historical dates and values.
            n_steps (int): Number of steps to predict.
            posterior_samples: Extracted posterior samples from arviz.
            **kwargs: Additional arguments.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Summary and samples DataFrames.
        """
        pass

    @abc.abstractmethod
    def _extract_priors(self, trace) -> dict:
        """
        Extract parameter estimates from fitted trace to create informed priors.

        This method should analyze the posterior distributions from model fitting
        and return a dictionary of updated prior parameters that can be used
        for subsequent fitting (e.g., in harmonized models).

        Args:
            trace: arviz.InferenceData containing the posterior samples

        Returns:
            dict: Dictionary of updated prior parameters specific to this model type
        """
        pass

    def _check_for_dates_and_values(self, dataset: pd.DataFrame):
        """
        Validate that dataset contains required 'date' and 'value' columns.

        Args:
            dataset (pd.DataFrame): Dataset to validate.

        Returns:
            tuple: (dates array, values array)

        Raises:
            ValueError: If required columns are missing.
        """
        if "date" not in dataset.columns or "value" not in dataset.columns:
            raise ValueError("Dataset must contain 'date' and 'value' columns.")

        return dataset["date"].values, dataset["value"].values

    def _add_param_name_prefix(
        self, param_name: str, force_original_name: bool = False
    ) -> str:
        """
        Add model name prefix to parameter names to avoid conflicts.

        Args:
            param_name (str): Base parameter name.

        Returns:
            str: Prefixed parameter name in format "{model_name}_{param_name}".
        """
        if force_original_name:
            name = self.name
        else:
            name = self.cloned_from or self.name
        return f"{name}_{param_name}"

    def set_priors(self, priors: dict, **kwargs):
        """
        Set the priors for the model parameters.

        Args:
            priors (dict): Dictionary of prior parameters to set.
        """
        self.priors.update(priors)

    def post_fit(self, trace, **kwargs):
        """
        Hook for any post-processing after fitting the model.

        This method can be overridden by subclasses to perform any
        additional steps required after the model has been fitted.
        """
        pass
