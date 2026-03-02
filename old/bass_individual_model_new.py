from math import log
import pymc as pm
import pandas as pd
import pytensor.tensor as pt
import numpy as np
import arviz as az
from hierarchical.individual_model import SingleModel
from hierarchical.logging_config import get_logger
from rich import print

logger = get_logger(__name__)


def beta_mean_and_std_to_alpha_beta(mean: float, std: float):
    """
    Convert mean and standard deviation to alpha and beta parameters of a Beta distribution.

    Args:
        mean (float): Mean of the Beta distribution (0 < mean < 1).
        std (float): Standard deviation of the Beta distribution.
    Returns:
        tuple: (alpha, beta) parameters of the Beta distribution.
    """

    variance = std**2
    common_factor = (mean * (1 - mean) / variance) - 1
    alpha = mean * common_factor
    beta = (1 - mean) * common_factor
    return alpha, beta


def beta_alpha_and_beta_to_mean_std(alpha: float, beta: float):
    """
    Convert alpha and beta parameters of a Beta distribution to mean and standard deviation.

    Args:
        alpha (float): Alpha parameter of the Beta distribution.
        beta (float): Beta parameter of the Beta distribution.
    Returns:
        tuple: (mean, std) of the Beta distribution.
    """
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
    std = np.sqrt(variance)
    return mean, std


r_defaults_alpha_beta = {
    "r1": beta_mean_and_std_to_alpha_beta(0.189, 0.058),
    "r2": beta_mean_and_std_to_alpha_beta(0.051, 0.017),
    "r3": beta_mean_and_std_to_alpha_beta(0.344, 0.078),
}


class BassIndividualModel(SingleModel):
    """
    Bass Diffusion Model implementation for individual products.

    This model implements the classic Bass diffusion model with optional seasonal effects,
    suitable for modeling product adoption over time with innovation and imitation effects.

    The model structure is:
    y[t] = Bass_cumulative[t] * seasonal[season[t]] + noise[t]

    Where Bass_cumulative[t] = M * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))

    Attributes:
        include_seasonality (bool): Whether to include seasonal effects.
        n_seasons (int): Number of seasonal periods (e.g., 12 for monthly, 4 for quarterly).
    """

    def __init__(
        self,
        model: pm.Model,
        name: str,
        include_seasonality: bool = True,
        n_seasons: int = 4,
        adjustments: dict = {},
        should_fit: bool = True,
        cloned_from: str = None,
        seasonal_concentration_multiplier: float = 1.0,
        priors: dict = {},
        learning_inertia: float = 1.0,
        return_to_basic_model: bool = False,
        **kwargs,
    ):
        """
        Initialize the Bass Individual Model.

        Args:
            model (pm.Model): PyMC model instance.
            name (str): Unique name for this model instance.
            include_seasonality (bool, optional): Whether to include seasonal effects. Defaults to True.
            n_seasons (int, optional): Number of seasonal periods. Defaults to 4 (quarterly).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(model, name, **kwargs)
        self.include_seasonality = include_seasonality
        self.n_seasons = n_seasons
        self.should_fit = should_fit
        self.cloned_from = cloned_from
        self.adjustments = adjustments
        self.seasonal_concentration_multiplier = seasonal_concentration_multiplier
        self.eol_time = 12 * 4
        self.learning_inertia = learning_inertia
        self.return_to_basic_model = return_to_basic_model
        logger.debug("Inertia at init:", learning_inertia)

        beta_variables = ["p", "p2", "p3", "q", "q2", "r1", "r2", "r3"]

        self.priors = {}
        for var in beta_variables:
            alpha, beta = beta_mean_and_std_to_alpha_beta(0.1, 0.05)
            self.priors[f"{var}_alpha"] = alpha
            self.priors[f"{var}_beta"] = beta

        # self.priors["r2_alpha"], self.priors["r2_beta"] = (
        #     beta_mean_and_std_to_alpha_beta(0.1, 0.2)
        # )

        for var in ["r1", "r2", "r3"]:
            v = r_defaults_alpha_beta[var]
            adjusted = [v[0], v[1] * 3]
            self.priors[f"{var}_alpha"], self.priors[f"{var}_beta"] = adjusted

        self.priors = self.priors | {
            "M_mu": 5,
            "M_sigma": 0.5,
            "seasonal_concentration": 1.0,
            **self.params.get("priors", {}),
        }

        self.priors = {**self.priors, **priors}
        logger.debug("Applying inertia:", learning_inertia)
        self.learning_inertia = learning_inertia
        if learning_inertia > 1:
            logger.debug("Adjusting priors with inertia:", learning_inertia)
            for key in beta_variables:
                alpha = self.priors[f"{key}_alpha"]
                beta = self.priors[f"{key}_beta"]
                mean, std = beta_alpha_and_beta_to_mean_std(alpha, beta)
                logger.debug(f"Adjusting {key} prior: mean={mean}, std={std}")
                adjusted_std = std / learning_inertia
                new_alpha, new_beta = beta_mean_and_std_to_alpha_beta(
                    mean, adjusted_std
                )
                logger.debug(f"New {key} prior: alpha={new_alpha}, beta={new_beta}")
                self.priors[f"{key}_alpha"] = new_alpha
                self.priors[f"{key}_beta"] = new_beta

            self.priors["M_sigma"] = self.priors["M_sigma"] / learning_inertia

    def forward(
        self, dataset: pd.DataFrame, readjust_parameters: bool = True, **kwargs
    ):
        """
        Define the Bass diffusion model structure within the PyMC model context.

        Creates the following model components:
        - Innovation coefficient (p): Beta prior for external influence
        - Imitation coefficient (q): Beta prior for word-of-mouth effect
        - Market potential (M): LogNormal prior for total addressable market
        - Launch time (t0): Automatically detected or estimated from data
        - Seasonal effects (optional): Dirichlet-distributed multiplicative seasonal adjustments
        - Noise variance (sigma): Half-normal prior for observation noise
        - Likelihood: Normal observation model with Bass cumulative function starting from t0

        Args:
            dataset (pd.DataFrame): Training dataset with 'date' and 'value' columns.
                                  Values can be incremental (will be converted to cumulative)
            **kwargs: Additional model specification arguments.
        """
        dates, y = self._check_for_dates_and_values(dataset)
        t0_idx = self._detect_launch_time(y)
        time_points = self._get_time_points(dates)

        # Store t0 information for the model
        self.t0_idx = t0_idx
        self.t0_time = time_points[t0_idx] if t0_idx < len(time_points) else 0.0

        # Detect if data is incremental vs cumulative
        is_incremental = self._detect_data_type(y)

        # Store original data for likelihood
        if is_incremental:
            logger.info("Detected incremental data, will model as Bass incremental")
            y_observed = y.copy()  # Keep incremental for likelihood
            y_cumulative = np.cumsum(y)  # For reference only
        else:
            logger.info("Detected cumulative data, using as-is")
            y_observed = y.copy()
            y_cumulative = y.copy()

            raise NotImplementedError(
                "Cumulative data handling not implemented yet. Please provide incremental data."
            )

        logger.info(f"Using launch time index t0 = {t0_idx}")
        logger.info(
            f"Launch date = {dates[t0_idx] if t0_idx < len(dates) else 'Unknown'}"
        )

        time_points = self._get_time_points(dates)
        time_since_launch = pt.maximum(0, time_points - self.t0_time)

        # Adjust M prior based on data (remember M_mu is on log scale for LogNormal)

        if readjust_parameters:
            data_max_estimate = y_cumulative.max() * 1.1
            current_M_estimate = np.exp(self.priors["M_mu"])

            if data_max_estimate > current_M_estimate:
                logger.info(
                    f"Readjusting M prior based on data: {data_max_estimate:.1f} (was {current_M_estimate:.1f})"
                )
                self.priors["M_mu"] = np.log(data_max_estimate)
                self.priors["M_sigma"] = 0.5  # Keep reasonable uncertainty on log scale

            max_time_since_launch = np.max(time_since_launch.eval())
            if max_time_since_launch < self.eol_time:
                for var in ["r1", "r2", "r3"]:
                    self.priors[f"{var}_alpha"], self.priors[f"{var}_beta"] = (
                        r_defaults_alpha_beta[var]
                    )

                logger.info(
                    f"Data does not extend beyond end-of-life time of {self.eol_time}, adjusting p3 and r2 priors to near-zero."
                )

        with self.model:
            # Bass diffusion parameters
            # Innovation coefficient (external influence)
            if self.should_fit == False:
                self.pymc_params["mu"] = pm.Data(
                    self._add_param_name_prefix("mu"), y_observed
                )

                return
                # self.pymc_params['mu'] = pm.Data

            beta_vars = ["p", "p2", "p3", "q", "q2", "r1", "r2", "r3"]

            for var in beta_vars:
                self.pymc_params[var] = pm.Beta(
                    self._add_param_name_prefix(var),
                    alpha=self.priors.get(f"{var}_alpha", 1.0),
                    beta=self.priors.get(f"{var}_beta", 1.0),
                )

            time_to_peak_mu = self.priors.get("time_to_peak_adoption_mu")
            time_to_peak_sigma = self.priors.get(
                "time_to_peak_adoption_sigma",
                self.priors.get("time_to_peak_adoption_std"),
            )
            if time_to_peak_mu is not None and time_to_peak_sigma is not None:
                time_to_peak = (
                    pt.log(self.pymc_params["q"]) - pt.log(self.pymc_params["p"])
                ) / (self.pymc_params["p"] + self.pymc_params["q"])
                pm.Potential(
                    self._add_param_name_prefix("time_to_peak_adoption_potential"),
                    pm.logp(
                        pm.Normal.dist(
                            mu=time_to_peak_mu, sigma=time_to_peak_sigma
                        ),
                        time_to_peak,
                    ),
                )

            # Market potential
            logger.debug(f"Model name: {self.name}")
            logger.debug("Setting M prior with mu:", self.priors["M_mu"])
            logger.debug("Setting M prior with sigma:", self.priors["M_sigma"])
            self.pymc_params["M"] = pm.LogNormal(
                self._add_param_name_prefix("M"),
                mu=self.priors["M_mu"],
                sigma=self.priors["M_sigma"],
            )
            # Convert dates to time periods (normalized)

            # Store t0 information for the model
            self.t0_idx = t0_idx
            self.t0_time = time_points[t0_idx] if t0_idx < len(time_points) else 0.0

            # Bass cumulative adoption function (time adjusted for launch)
            # For times before t0, adoption should be 0
            # For times after t0, use Bass diffusion starting from 0
            params = [self.pymc_params[var] for var in beta_vars] + [
                self.pymc_params["M"]
            ]

            bass_incremental = self._bass_incremental_from_diff_equation(
                params, y_cumulative, time_since_launch
            )

            # Set adoption to 0 before launch time
            pre_launch_mask = time_points < self.t0_time
            bass_incremental = pt.switch(pre_launch_mask, 0.0, bass_incremental)

            # Seasonal effects (if enabled)
            if self.include_seasonality:
                seasonal_indices = self._get_seasonal_indices(dates)

                if "seasonal_weights" in self.priors:
                    seasonal_priors = self.priors.get(
                        "seasonal_weights", np.ones(self.n_seasons)
                    )

                    assert len(seasonal_priors) == self.n_seasons, (
                        f"Length of seasonal_weights prior ({len(seasonal_priors)}) "
                        f"must match n_seasons ({self.n_seasons})"
                    )

                    seasonal_priors = np.array(seasonal_priors) / np.sum(
                        seasonal_priors
                    )

                else:
                    seasonal_priors = (
                        np.ones(self.n_seasons) * self.priors["seasonal_concentration"]
                    )

                # Seasonal factors using Dirichlet to ensure proper normalization
                seasonal_weights = pm.Dirichlet(
                    self._add_param_name_prefix("seasonal_weights"),
                    a=seasonal_priors * self.seasonal_concentration_multiplier,
                )
                seasonal_factors = (
                    seasonal_weights * self.n_seasons
                )  # Scale to sum to n_seasons
                self.pymc_params["seasonal_factors"] = seasonal_factors

                # Apply seasonal multipliers - ensure indices are numpy array
                seasonal_indices_array = np.array(seasonal_indices, dtype=int)
                seasonal_multipliers = seasonal_factors[seasonal_indices_array]
                # mu_cumulative = bass_cumulative
                factors = seasonal_multipliers
            else:
                # mu_cumulative = bass_cumulative
                factors = 1

            mu = bass_incremental * factors
            likelihood_data = y_observed  # Use original incremental data
            # else:
            #     mu = mu_cumulative
            #     likelihood_data = y_observed  # Use cumulative data

            # Store the final mean parameter
            self.pymc_params["mu"] = mu

            # Observation noise - scale appropriately for the type of data
            if is_incremental:
                # For incremental data, noise should be scaled to incremental values
                noise_scale = y_observed.std() * 0.2 if len(y_observed) > 1 else 1000.0
            else:
                # For cumulative data, noise should be scaled to cumulative values
                noise_scale = y_observed.std() * 0.1 if len(y_observed) > 1 else 10000.0

            sigma = pm.HalfNormal(
                self._add_param_name_prefix("sigma"), sigma=noise_scale
            )
            self.pymc_params["sigma"] = sigma

            # Likelihood with appropriate data
            self.likelihood = pm.Normal(
                self._add_param_name_prefix("y"),
                mu=mu,
                sigma=sigma,
                observed=likelihood_data,
            )

    def _detect_data_type(self, y):
        """
        Detect if data is incremental or cumulative.

        Args:
            y: Array of observed values

        Returns:
            bool: True if incremental, False if cumulative
        """
        # Remove zeros from the beginning

        return True

    def _detect_launch_time(self, y, threshold=1e-6):
        """
        Automatically detect the launch time (t0) from the data.

        Args:
            y: Array of observed values
            threshold: Minimum value to consider as "launched" (default: 0.01)

        Returns:
            Index of the first significant value (launch time)
        """
        # Find first non-zero or significant value
        nonzero_indices = np.where(y > threshold)[0]

        if len(nonzero_indices) > 0:
            t0_idx = nonzero_indices[0]
            logger.debug(
                f"Detected launch time at index {t0_idx} (value: {y[t0_idx]:.3f})"
            )
        else:
            # If no significant values found, assume launch at the very end
            t0_idx = len(y) - 1
            logger.warning(
                f"No clear launch point detected, assuming launch at end (value: {y[t0_idx]:.3f})"
            )

        return t0_idx

    def _bass_incremental_from_diff_equation(
        self, params: tuple, cumulative_data: np.ndarray, time_since_launch: np.ndarray
    ):
        """Return Bass incremental adoption using the differential equation form.

        remaining_market = ( M - S(t-t) )
        adoption_rate = (
            p +
            p3 * ( 1 - exp(-p2*t)) +
            q * (S(t-t) / M) +
            q2 * (S(t-t) / M)^2
        )
        incremental = remaining_market * adoption_rate

        +

        Args:
            p: Innovation coefficient
            q: Imitation coefficient
            M: Market potential
            cumulative_data: Array of cumulative adoption values
            time_since_launch: Array of time points since launch
        """
        p, p2, p3, q, q2, r1, r2, r3, M = params
        # Get the last n-1 cumulative values (we need S(t-1) for each time step)
        lagged_cumulative = pt.concatenate(
            [
                [0],  # S(0) = 0 for the first period
                cumulative_data[:-1],  # S(t-1) for t=1,2,...,n
            ]
        )

        has_launched = time_since_launch > 0

        use_basic_model = self.return_to_basic_model
        additional_terms_multiplier = ( 1 - int(use_basic_model))

        exp_term = p3 * (1 - pt.exp(-p2 * time_since_launch)) 
        remaining_market = pt.maximum(M - lagged_cumulative, 1e-9)

        is_going_to_eol = 1 - pt.exp(
            -r1 * pt.maximum(0, time_since_launch - self.eol_time)  # R1
        )
        eol_factor = r2 * is_going_to_eol  # R2
        residual_remaining_market = pt.maximum(
            M * (1 + r3) - lagged_cumulative, 1e-9
        )   # R3
        adoption_rate = (
            p
            + exp_term * additional_terms_multiplier
            + q * (lagged_cumulative / M)
            + q2 * pt.square(lagged_cumulative / M) * additional_terms_multiplier
        )
        incremental = (
            remaining_market * adoption_rate * has_launched
            + eol_factor * residual_remaining_market * additional_terms_multiplier
        )

        return incremental

    def _bass_incremental_from_diff_equation_numpy(
        self, params, initial_sum: float, time_since_launch: np.ndarray
    ):
        """
        Numpy version of Bass incremental adoption using the differential equation form.

        This method now properly matches the PyTensor version by using lagged cumulative values.

        remaining_market = ( M - S(t-1) )
        remaining_market_after_end_of_life = ( M*(1+r3) - S(t-1) )
        end_of_life_factor = r2 * (1 - exp(-r1 * max(0, t - t0 - 12*4)))
        adoption_rate = (
            p +
            p3 * ( 1 - exp(-p2*t)) +
            q * (S(t-1) / M) +
            q2 * (S(t-1) / M)^2
        )
        incremental = remaining_market * adoption_rate  + end_of_life_factor * remaining_market_after_end_of_life

        Args:
            params: Tuple of (p, p2, p3, q, q2, r1, r2, M) parameters
            initial_sum: Starting cumulative value (this is S(t-1) for the first prediction)
            time_since_launch: Array of time points since launch

        Returns:
            Array of incremental adoption values
        """
        p, p2, p3, q, q2, r1, r2, r3, M = params

        forecasts = np.zeros_like(time_since_launch) * 0.0

        # Initialize with the lagged cumulative values
        # For the first prediction, we use initial_sum as S(t-1)
        lagged_cumulative = initial_sum

        for idx, t in enumerate(time_since_launch):
            if t <= 0:  # Before launch
                forecasts[idx] = 0.0
                # For next iteration, lagged_cumulative stays the same since we added 0
                continue
            use_basic_model = self.return_to_basic_model
            additional_terms_multiplier = 1 - int(use_basic_model)

            # End of life factor
            eol_factor = (
                r2
                * (1 - np.exp(-r1 * max(0, t - self.eol_time)))
                * additional_terms_multiplier
            )

            # Extended Bass adoption rate components
            exp_term = (
                p3 * (1 - np.exp(-p2 * t)) * additional_terms_multiplier
            )
            # print(exp_term)

            # Use lagged cumulative (S(t-1)) just like the PyTensor version
            remaining_market = max(M - lagged_cumulative, 1e-9)

            residual_remaining_market = max(M * (1 + r3) - lagged_cumulative, 1e-9)

            # Extended adoption rate: p + p3*(1-exp(-p2*t)) + q*S(t-1)/M + q2*(S(t-1)/M)^2
            adoption_rate = (
                p
                + exp_term
                + q * (lagged_cumulative / M)
                + q2 * (lagged_cumulative / M) ** 2 * additional_terms_multiplier
            )

            # Incremental adoption with end of life factor
            incremental = (
                remaining_market * adoption_rate
                + eol_factor * residual_remaining_market
            )

            # Ensure non-negative
            # incremental = max(0, incremental)

            forecasts[idx] = incremental

            # Update lagged_cumulative for next iteration: S(t) = S(t-1) + incremental(t)
            lagged_cumulative += incremental

        return forecasts

    def _get_time_points(self, dates):
        """
        Convert dates to normalized time points for the Bass model.

        Args:
            dates: Array of dates

        Returns:
            Normalized time points starting from 0
        """
        if pd.api.types.is_datetime64_any_dtype(dates):
            # Convert to days from start, then normalize to roughly monthly units
            start_date = pd.to_datetime(dates[0])
            days_from_start = (pd.to_datetime(dates) - start_date).days.values
            return days_from_start / 30.44  # Convert to approximate months
        else:
            # Assume already numeric time points
            return dates - dates[0]

    def _get_seasonal_indices(self, dates):
        """
        Convert dates to seasonal indices for the model.

        Args:
            dates: Array of dates (datetime or numeric).

        Returns:
            np.ndarray: Array of seasonal indices (0 to n_seasons-1).
        """
        if pd.api.types.is_datetime64_any_dtype(dates):
            indices = (pd.to_datetime(dates).month - 1) % self.n_seasons
        #     if self.n_seasons == 12:
        #         # Monthly seasonality
        #         indices = (pd.to_datetime(dates).month - 1) % 12
        #     elif self.n_seasons == 4:
        #         # Quarterly seasonality
        #         indices = (pd.to_datetime(dates).quarter - 1) % 4
        #     else:
        #         # Generic seasonal mapping
        #         indices = np.arange(len(dates)) % self.n_seasons
        # else:
        #     # For numeric data, assume sequential mapping to seasons
        #     indices = np.arange(len(dates)) % self.n_seasons

        # Ensure we return a numpy array, not pandas Index, and clip to valid range
        indices = np.array(indices, dtype=int)
        # Safety check: ensure all indices are within valid range [0, n_seasons-1]
        indices = np.clip(indices, 0, self.n_seasons - 1)

        return indices

    def adjust(self, adjustments: dict):
        """
        Apply adjustments to model parameters based on hierarchical pretraining.

        Args:
            adjustments (dict): Dictionary of parameter adjustments.
                                Keys can include 'p', 'q', 'M', 'sigma', 't0'.
                                Values are multiplicative factors for p, q, M, sigma,
                                and additive for t0 (in time units).
        """
        self.adjustments = adjustments

    def _generate_predictions(
        self,
        dataset,
        n_steps,
        posterior_samples,
        adjustments: dict = {},
        seed: int = 42,
        **kwargs,
    ):
        """
        Generate predictions for future time steps using posterior samples.

        For each posterior sample, this method:
        1. Extracts Bass diffusion parameters
        2. Generates future time points
        3. Calculates Bass cumulative adoption for future periods
        4. Applies seasonal effects if enabled
        5. Adds observation noise

        Args:
            dates: Array of historical dates.
            y: Array of historical values.
            n_steps (int): Number of future steps to predict.
            posterior_samples: Extracted posterior samples from arviz.
            **kwargs: Additional prediction arguments.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - Summary statistics (mean, median, HDI)
                - Full posterior predictive samples
        """

        dates, y = self._check_for_dates_and_values(dataset)

        base_sum = np.sum(y)

        np.random.seed(seed)
        # Generate future dates and time points
        future_dates = self._generate_future_dates(dates, n_steps)
        # print(f"Future dates: {future_dates}")
        all_time_points = self._get_time_points(np.concatenate([dates, future_dates]))
        # print(f"All time points: {all_time_points}")
        future_time_points = all_time_points[-n_steps:]  # Only future points

        t0_idx = self._detect_launch_time(y)
        t0_time = all_time_points[t0_idx] if t0_idx < len(all_time_points) else 0.0

        # Adjust time points for launch time
        t0_time = t0_time + self.adjustments.get("t0", 0.0)
        future_time_since_launch = np.maximum(0, future_time_points - t0_time)

        full_adjustments = self.adjustments | adjustments

        # Parameter names for the Bass model
        parameter_names = ["p", "p2", "p3", "q", "q2", "r1", "r2", "r3", "M", "sigma"]

        # Extract parameter samples using dictionary approach
        samples = {}
        for param in parameter_names:
            param_key = self._add_param_name_prefix(param)
            if param_key in posterior_samples:
                samples[param] = posterior_samples[
                    param_key
                ].values * full_adjustments.get(param, 1.0)

        # Get seasonal information if needed
        if self.include_seasonality:
            # Note: The parameter in the trace is 'seasonal_weights', not 'seasonal_factors'
            # We need to apply the same transformation: seasonal_factors = seasonal_weights * n_seasons
            seasonal_weights_samples = posterior_samples[
                self._add_param_name_prefix("seasonal_weights")
            ].values
            # Apply the same transformation as in forward method
            seasonal_factors_samples = seasonal_weights_samples * self.n_seasons

            future_seasonal_indices = self._get_seasonal_indices(future_dates)
            future_seasonal_indices = np.array(future_seasonal_indices, dtype=int)

        # Generate predictions for each posterior sample
        predictions = np.zeros((len(samples["p"]), n_steps))

        for i in range(len(samples["p"])):
            # Calculate Bass cumulative adoption (adjusted for launch time)
            params = (
                samples["p"][i],
                samples["p2"][i],
                samples["p3"][i],
                samples["q"][i],
                samples["q2"][i],
                samples["r1"][i],
                samples["r2"][i],
                samples["r3"][i],
                samples["M"][i],
            )
            S = base_sum

            bass_incremental = self._bass_incremental_from_diff_equation_numpy(
                params,
                base_sum,
                future_time_since_launch,
            )

            # Apply seasonal effects if enabled
            if self.include_seasonality:
                # Safety check for indices
                safe_indices = np.clip(
                    future_seasonal_indices, 0, seasonal_factors_samples.shape[0] - 1
                )

                # Extract the seasonal multipliers for this sample
                # seasonal_factors_samples shape is (n_seasons, n_samples)
                seasonal_multipliers = seasonal_factors_samples[safe_indices, i]
                bass_incremental = bass_incremental * seasonal_multipliers

            # Add noise and store predictions
            for t in range(n_steps):
                pred_t = np.random.normal(bass_incremental[t], samples["sigma"][i])
                # Ensure predictions are non-negative (can't have negative adoption)
                pred_t = max(0, pred_t)
                predictions[i, t] = pred_t

        # Create summary DataFrame
        summary_df = pd.DataFrame(
            {
                "date": future_dates,
                "mean": np.mean(predictions, axis=0),
                "median": np.median(predictions, axis=0),
                "hdi_5%": np.percentile(predictions, 5, axis=0),
                "hdi_95%": np.percentile(predictions, 95, axis=0),
            }
        )

        # Create samples DataFrame
        samples_list = []
        for i in range(len(predictions)):
            for t in range(n_steps):
                samples_list.append(
                    {
                        "date": future_dates[t],
                        "value": predictions[i, t],
                        "sample_id": i,
                    }
                )

        samples_df = pd.DataFrame(samples_list)

        return summary_df, samples_df

    def _extract_priors(self, trace) -> dict:
        """
        Extract Bass model parameter estimates from fitted trace to create informed priors.

        Args:
            trace: arviz.InferenceData containing the posterior samples

        Returns:
            dict: Dictionary of updated prior parameters for Bass model
        """
        updated_priors = {}

        try:
            values = az.extract(trace, "posterior", num_samples=1000)

            beta_variables = ["p", "p2", "p3", "q", "q2", "r1", "r2", "r3"]

            for var in beta_variables:
                param_name = f"{self.name}_{var}"
                if param_name in values:
                    samples = values[param_name].values

                    assert samples.shape[0] > 0, f"No samples found for variable {var}"

                    mean_sample = np.mean(samples)
                    std_sample = np.std(samples)

                    # Convert to alpha and beta parameters
                    alpha, beta = beta_mean_and_std_to_alpha_beta(
                        mean_sample, std_sample
                    )

                    updated_priors[f"{var}_alpha"] = alpha
                    updated_priors[f"{var}_beta"] = beta
                else:
                    logger.warning(
                        f"Parameter {param_name} not found in trace, skipping prior extraction"
                    )

            # Market potential (M) - convert to LogNormal parameters
            M_samples = values[f"{self.name}_M"].values
            M_log_samples = np.log(M_samples[M_samples > 0])
            if len(M_log_samples) > 0:
                updated_priors.update(
                    {
                        "M_mu": np.mean(M_log_samples),
                        "M_sigma": np.std(M_log_samples),
                    }
                )

            # Seasonal concentration (if applicable)
            if True:  # self.include_seasonality:
                # try:
                # Use 'seasonal_weights' not 'seasonal_factors'
                seasonal_weights_samples = values[
                    f"{self.name}_seasonal_weights"
                ].values
                # Apply the same transformation as in forward method to get seasonal_factors
                seasonal_factors_samples = seasonal_weights_samples * self.n_seasons
                # Use coefficient of variation to set concentration
                cv = np.std(seasonal_factors_samples) / np.mean(
                    seasonal_factors_samples
                )
                # Higher concentration = lower variability
                concentration = max(0.5, 1.0 / cv)
                updated_priors["seasonal_concentration"] = concentration
                updated_priors["seasonal_weights"] = seasonal_weights_samples.mean(
                    axis=1
                )

        except Exception as e:
            logger.error(f"Error extracting Bass priors for {self.name}: {e}")

        return updated_priors

    def _bass_cumulative_numpy(self, t, p, q, M):
        """
        Numpy version of Bass cumulative function for predictions.

        Args:
            t: Time points array
            p: Innovation coefficient
            q: Imitation coefficient
            M: Market potential

        Returns:
            Cumulative adoption array
        """
        if p == 0:
            return M * q * t

        denom = p + q
        exp_term = np.exp(-denom * t)
        numerator = M * (1 - exp_term)
        denominator = 1 + (q / p) * exp_term
        return numerator / denominator

    def _generate_future_dates(self, dates, n_steps):
        """
        Generate future dates based on the pattern in historical dates.

        Args:
            dates: Historical dates array.
            n_steps: Number of future steps.

        Returns:
            Array of future dates.
        """
        if pd.api.types.is_datetime64_any_dtype(dates):
            # For datetime data, infer frequency
            last_date = pd.to_datetime(dates[-1])
            try:
                freq = pd.infer_freq(pd.to_datetime(dates[-min(10, len(dates)) :]))
                if freq is None:
                    # Fallback to monthly if frequency cannot be inferred
                    freq = "M"
                future_dates = pd.date_range(
                    start=last_date, periods=n_steps + 1, freq=freq
                )[1:]
            except:
                # Fallback for irregular datetime data
                avg_diff = np.mean(np.diff(pd.to_datetime(dates[-5:])))
                future_dates = [last_date + i * avg_diff for i in range(1, n_steps + 1)]
                future_dates = pd.to_datetime(future_dates)

            return future_dates.values
        else:
            # For numeric data, assume regular intervals
            return np.arange(dates[-1] + 1, dates[-1] + 1 + n_steps)

    def post_fit(self, trace, **kwargs):
        """
        Hook for any post-processing after fitting the model.

        This method can be overridden by subclasses to perform any
        additional steps required after the model has been fitted.
        """
        pass

    def set_priors(self, priors: dict, ignore_inertia: bool = False, **kwargs):
        """
        Set the priors for the model parameters.

        Args:
            priors (dict): Dictionary of prior parameters to set.
        """

        beta_variables = ["p", "p2", "p3", "q", "q2", "r1", "r2", "r3"]

        if self.learning_inertia > 1 and not ignore_inertia:
            logger.debug("Adjusting priors with inertia:", self.learning_inertia)
            for key in beta_variables:
                alpha = priors[f"{key}_alpha"]
                beta = priors[f"{key}_beta"]
                mean, std = beta_alpha_and_beta_to_mean_std(alpha, beta)
                logger.debug(f"Adjusting {key} prior: mean={mean}, std={std}")
                adjusted_std = std / self.learning_inertia
                new_alpha, new_beta = beta_mean_and_std_to_alpha_beta(
                    mean, adjusted_std
                )
                logger.debug(f"New {key} prior: alpha={new_alpha}, beta={new_beta}")
                priors[f"{key}_alpha"] = new_alpha
                priors[f"{key}_beta"] = new_beta

            priors["M_sigma"] = priors["M_sigma"] / self.learning_inertia

        self.priors.update(priors)
