import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from .logging_config import get_logger

logger = get_logger(__name__)


class BaseHarmonizer(ABC):
    """
    Base class for post-hoc harmonization of hierarchical forecasting predictions.

    This class provides a common interface for different harmonization methods
    that reconcile predictions to satisfy hierarchical constraints.
    """

    def __init__(self, dependencies: Dict[str, List[List[str]]], **kwargs):
        """
        Initialize the harmonizer.

        Args:
            dependencies: Dictionary mapping parent models to lists of child groups.
                         Format: {parent: [[child1, child2], [child3, child4]]}
            **kwargs: Additional parameters for specific harmonizers
        """
        self.dependencies = dependencies
        self.validate_dependencies()

    def validate_dependencies(self):
        """Validate the structure of the dependencies dictionary."""
        if not isinstance(self.dependencies, dict):
            raise ValueError("Dependencies must be a dictionary")

        for parent, child_groups in self.dependencies.items():
            if not isinstance(child_groups, list):
                raise ValueError(f"Child groups for {parent} must be a list")
            for group in child_groups:
                if not isinstance(group, list):
                    raise ValueError(f"Each child group must be a list of strings")

    @abstractmethod
    def harmonize(
        self, data: Dict[str, List[np.ndarray]], **kwargs
    ) -> Dict[str, List[np.ndarray]]:
        """
        Harmonize predictions to satisfy hierarchical constraints.

        Args:
            predictions_dict: Dictionary mapping model names to lists of prediction arrays
            **kwargs: Additional method-specific parameters

        Returns:
            Dictionary mapping model names to harmonized prediction arrays
        """
        pass

    def validate_predictions(self, predictions_dict: Dict[str, List[np.ndarray]]):
        """Validate the structure of predictions dictionary."""
        if not predictions_dict:
            raise ValueError("Predictions dictionary cannot be empty")

        # Check that all predictions have the same number of samples
        sample_counts = [len(samples) for samples in predictions_dict.values()]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                "All models must have the same number of prediction samples"
            )

        # Check that all prediction arrays have the same length (time steps)
        first_model = next(iter(predictions_dict.keys()))
        n_steps = len(predictions_dict[first_model][0])

        for model_name, samples in predictions_dict.items():
            for i, sample in enumerate(samples):
                if len(sample) != n_steps:
                    raise ValueError(
                        f"All prediction samples must have the same length. "
                        f"Model {model_name}, sample {i} has length {len(sample)}, "
                        f"expected {n_steps}"
                    )

    def evaluate_results(
        self, results: pd.DataFrame, original_data: dict[List[dict]]
    ) -> dict[str, float]:
        """Evaluate the results of the harmonization"""

        averages = results.groupby(["t", "variable"])["value"].mean().unstack(level=1)

        for parent in self.dependencies:
            parent_data = averages[parent]
            groups = self.dependencies[parent]

            for idx, group in enumerate(groups):
                children_sum = averages[group].sum(axis=1)

                original_error = np.zeros_like(children_sum)
                # print(original_error)

                for t in range(len(original_error)):
                    for child in group:
                        original_error[t] += original_data[child][t]["mu"]

                    original_error[t] -= original_data[parent][t]["mu"]

                original_error = np.abs(original_error)
                averages[f"{parent}_{idx}_error"] = np.abs(parent_data - children_sum)
                averages[f"{parent}_{idx}_error_original"] = original_error

        return averages


class pyMCHarmonizer(BaseHarmonizer):
    def __init__(self, dependencies: Dict[str, List[List[str]]]):
        super().__init__(dependencies=dependencies)
        self.model = pm.Model()

    def _numpy_samples_to_pandas(self, samples: dict[str, np.ndarray]) -> pd.DataFrame:
        df = pd.DataFrame()

        for sample_name in samples:
            data = samples[sample_name].values
            this_df = pd.DataFrame(data).stack().to_frame().reset_index()
            this_df.columns = ["t", "sample_id", "value"]
            this_df["variable"] = sample_name

            df = pd.concat([df, this_df], axis=0)

        return df

    def harmonize(
        self,
        data: Dict[str, List[dict[str, float]]],
        lambda_harmonization: float = 1,
        pymc_kwargs: dict = {"draws": 1000, "samples": 1000, "chains": 2},
        rng: int = 42,
    ) -> Dict[str, List[np.ndarray]]:

        dependencies = self.dependencies

        with self.model:
            factors = {}

            for model_name in data:
                ts = data[model_name]
                mu = [x["mu"] for x in ts]
                std = [x["std"] for x in ts]
                factors[model_name] = pm.Normal(
                    model_name, mu=mu, sigma=std, shape=len(ts)
                )

            constraints = {}
            for top_level in dependencies:
                ts = data[top_level]
                stds = [x["std"] for x in ts]
                median_std = np.median(stds)
                for idx, group in enumerate(dependencies[top_level]):
                    top_level_factor = factors[top_level]
                    bottom_factors = []
                    for elm in group:
                        bottom_factors.append(factors[elm])

                    sum_children = pt.stack(bottom_factors).sum(axis=0)

                    constraints[top_level + str(idx)] = pm.Potential(
                        f"harmonization_{top_level}_{idx}",
                        -lambda_harmonization
                        * ((sum_children - top_level_factor) / median_std) ** 2,
                    )

            trace = pm.sample(**pymc_kwargs)

        posterior_samples = az.extract(trace, num_samples=1000, rng=rng)

        return self._numpy_samples_to_pandas(posterior_samples).set_index(
            "t", drop=True
        )


class PriceHarmonizer(BaseHarmonizer):
    def __init__(
        self,
        dependencies: Dict[str, List[List[str]]],
        dollar_suffix: str = "_dollar",
        qty_suffix: str = "_qty",
    ):
        super().__init__(dependencies=dependencies)
        self.dollar_suffix = dollar_suffix
        self.qty_suffix = qty_suffix
        self.model = pm.Model()

    def _numpy_samples_to_pandas(self, samples: dict[str, np.ndarray]) -> pd.DataFrame:
        df = pd.DataFrame()

        for sample_name in samples:
            data = samples[sample_name].values
            this_df = pd.DataFrame(data).stack().to_frame().reset_index()
            this_df.columns = ["t", "sample_id", "value"]
            this_df["variable"] = sample_name

            df = pd.concat([df, this_df], axis=0)

        return df

    def harmonize(
        self,
        data: Dict[str, List[dict[str, float]]],
        asp: Dict[str, List[dict[str, float]]],
        lambda_harmonization: float = 1,
        lambda_price: float = 1,
        pymc_kwargs: dict = {"draws": 1000, "samples": 1000, "chains": 2},
        beliefs: Optional[Dict[str, float]] = {},
        rng: int = 42,
    ) -> Dict[str, List[np.ndarray]]:
        dependencies = self.dependencies

        suffixes = [self.dollar_suffix, self.qty_suffix]

        with self.model:
            factors = {}

            full_stds = {}

            for model_name in data:
                logger.debug(f"Processing model: {model_name}")
                ts = data[model_name]
                mu = [x["mu"] for x in ts]
                belief = beliefs.get(model_name, 1)
                std = [x["std"] / (belief + 1e-6) for x in ts]

                full_stds[model_name] = np.median(std)
                factors[model_name] = pm.Normal(
                    model_name, mu=mu, sigma=std, shape=len(ts)
                )

            # print(factors)

            constraints = {}
            for top_level in dependencies:
                for suffix in suffixes:
                    ts = data[top_level + suffix]
                    stds = [x["std"] for x in ts]
                    median_std = np.median(stds)
                    for idx, group in enumerate(dependencies[top_level]):
                        top_level_factor = factors[top_level + suffix]
                        bottom_factors = []
                        for elm in group:
                            bottom_factors.append(factors[elm + suffix])

                        sum_children = pt.stack(bottom_factors).sum(axis=0)

                        constraints[top_level + str(idx)] = pm.Potential(
                            f"harmonization_{top_level}_{idx}_{suffix}",
                            -lambda_harmonization
                            * ((sum_children - top_level_factor) / median_std) ** 2,
                        )

            for model_name in asp:
                ts = asp[model_name]
                mu = [x["mu"] for x in ts]
                belief = beliefs.get(model_name + "_asp", 1)
                std = [x["std"] / (belief + 1e-6) for x in ts]

                asp_name = model_name + "_asp"
                price_factor = pm.Normal(asp_name, mu=mu, sigma=std, shape=len(ts))
                quantity = factors[model_name + self.qty_suffix]
                dollar = factors[model_name + self.dollar_suffix]

                normalization = full_stds[model_name + self.dollar_suffix]

                constraints[asp_name] = pm.Potential(
                    f"asp_constraint_{model_name}",
                    -lambda_price
                    * (
                        (price_factor * quantity - dollar)  # Avoid division by zero
                        / normalization
                    )
                    ** 2,
                )

            trace = pm.sample(**pymc_kwargs)

        posterior_samples = az.extract(trace, num_samples=1000, rng=rng)

        return self._numpy_samples_to_pandas(posterior_samples).set_index(
            "t", drop=True
        )
