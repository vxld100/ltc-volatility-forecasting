import logging
from typing import Tuple
import scipy.linalg
import scipy.stats
import numpy as np
import math

float64 = np.float64

from decimal import ROUND_HALF_UP, Decimal, getcontext

from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(
    level=logging.ERROR,  # Set the logging level (INFO, DEBUG, WARNING, ERROR, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Add timestamp and log level
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)

getcontext().prec = 28

class KernelEstimator:
    def __init__(self, alpha: float = 4.0, beta: float = 4.0, theta: float = 5.0, gamma: float = 0.49) -> None:
        # Constants calculated by integration (see appendix of Thesis)
        self.Phi_11 = Decimal(14) / Decimal(27)
        self.Phi_12 = Decimal(247) / Decimal(7290)
        self.Phi_22 = Decimal(985) / Decimal(183708)


        # Alpha is the parameter influencing the truncation level
        self.alpha = Decimal(str(alpha))

        # Beta is the parameter that influences the initial guess of the bandwidth
        # A value of 4 was seen to be reasonably good in a simulation, and is hence chosen as the default
        self.beta = Decimal(str(beta))

        self.default_theta = Decimal(str(theta))
        self.theta : Decimal 

        # Using equation (2.14) from the paper with α around 4-5 and gamma at 0.49 as suggested
        self.gamma = Decimal(str(gamma))

        self.truncation_level : Decimal


        self.bandwidth : Decimal # float
        self.pre_averaging_window : int
        self.estimator_weight : np.float64
        self.noise_variance : Decimal

        self.kernel_values = np.ndarray
        self.kernel_zero_index : int

        self.averaging_weights : np.ndarray
        self.debiasing_weights : np.ndarray

    def init_parameters(self, log_returns):
        n = Decimal(len(log_returns))
        dt = Decimal(1) / n

        self.theta = self.default_theta
        
        self.pre_averaging_window = self.get_pre_averaging_window(dt)
        self.estimator_weight = self.get_estimator_weight()
        self.bandwidth = self.get_bandwidth(dt=dt)
        self.noise_variance = self.estimate_noise_variance(log_returns)
        self.truncation_level = self.get_truncation_level_bpv(log_returns, dt)
        logging.info("Initialized all parameters")
        logging.info(f"Pre-averaging window = {self.pre_averaging_window}")
        logging.info(f"Bandwidth = {self.bandwidth}")
        logging.info(f"Truncation level = {self.truncation_level}")
        logging.info(f"Estimator weight = {self.estimator_weight}")
        logging.info(f"Noise Variance = {self.noise_variance}")

        self.kernel_values = self.compute_kernel_values(n=n)
        self.debiasing_weights = self.compute_debiasing_weights()
        self.averaging_weights = self.compute_averaging_weights()
        logging.info("Pre-computed all kernel values, averaging and debiasing weigths")

    def kernel(self, x: Decimal):
        return Decimal(str(0.5)) * Decimal.exp(- abs(x / self.bandwidth)) / self.bandwidth

    def compute_kernel_values(self, n: Decimal) -> np.ndarray:
        n_used_values = int(n - self.pre_averaging_window + 1) # Convert to int for the loop
        kernel_values = []
        # We pre-compute the kernel values for each possible distance
        for i in range(n_used_values):
            kernel_values.append(self.kernel(Decimal(i)/n))

        reverse_values = kernel_values.copy()
        reverse_values.reverse()
        reverse_values = reverse_values[:-1]  # Remove the duplicate 0 distance value
        full_vector = reverse_values + kernel_values

        self.kernel_zero_index = n_used_values - 1  # Store where the zero is located. -1 due to 0 based indexing

        return np.array([float(val) for val in full_vector], dtype=np.float64)

    def g(self, x: Decimal):
        return min(Decimal(2) * x, Decimal(1) - x)

    def compute_debiasing_weights(self):
        weights = []

        for j in range(1, self.pre_averaging_window + 1):
            diff = self.g(Decimal(j)/Decimal(self.pre_averaging_window)) - self.g(Decimal(j-1)/Decimal(self.pre_averaging_window))
            weight = diff ** Decimal(2)
            weights.append(weight)

        return np.array([float(weight) for weight in weights], dtype=np.float64)

    def compute_averaging_weights(self) -> np.ndarray:
        weights = []

        for j in range(1, self.pre_averaging_window):
            weights.append(self.g(Decimal(j)/Decimal(self.pre_averaging_window)))

        return np.array([float(weight) for weight in weights], dtype=np.float64)
        

    def get_bandwidth(self, dt: Decimal) -> Decimal:
        """
        The bandwidth is defined as m_n⋅Δ_n in the paper, see Section 3.2 for context
        """
        bandwidth = self.beta * dt ** Decimal('0.25')
        return bandwidth

    def get_pre_averaging_window(self, dt: Decimal) -> int:
        """
        Implements equation (2.13)
        """
        raw_value = Decimal(1) / (self.theta * Decimal.sqrt(dt))
    
        # Round to the nearest integer
        return int(raw_value.quantize(Decimal('1'), rounding=ROUND_HALF_UP))

    def get_estimator_weight(self) -> float64:
        """
        Implements the first part of equation (2.16)
        """
        weight = Decimal(0)
        for i in range(1, int(self.pre_averaging_window) + 1):
            weight += (self.g(Decimal(i)/Decimal(self.pre_averaging_window))) ** Decimal(2)
        return float64(weight)

    def get_truncation_level_simple(self, dt : Decimal) -> Decimal:
        v_n = self.alpha * (Decimal(self.pre_averaging_window) * dt) ** self.gamma
        return v_n

    def get_truncation_level_bpv(self, log_returns:np.ndarray, dt: Decimal) -> Decimal:
        # Calculate BPV directly as specified in the paper
        abs_returns = np.abs(log_returns)
        products = abs_returns[:-1] * abs_returns[1:]
        bpv = Decimal(str((np.pi / 2) * np.sum(products)))
        
        # Follow Jacod and Todorov's recommendation
        v_n = self.alpha * bpv.sqrt() * dt ** self.gamma
        
        return v_n

    def get_truncation_level(self, log_returns: np.ndarray, dt: Decimal):
        """
        Set truncation level for jump detection based on bipower variation
        calculated on sparsely sampled data (5-min frequency).
        """
        # Find non-zero returns
        non_zero_indices = np.where(log_returns != 0)[0]
        
        # Use 5-minute intervals as approximate targets
        n = len(log_returns)
        seconds_per_sample = 300  # 5 minutes = 300 seconds
        
        # Generate sparse sampling indices
        sparse_indices = []
        current_target = 0
        
        while current_target < n:
            # Find the closest non-zero return to the current target
            #closest_idx = non_zero_indices[np.argmin(np.abs(non_zero_indices - current_target))]
            sparse_indices.append(current_target + seconds_per_sample)
            current_target += seconds_per_sample
        
        # Get the sparse returns using the identified indices
        sparse_returns = log_returns.iloc[sparse_indices]
        
        # Calculate BPV on sparse returns
        abs_returns = np.abs(sparse_returns)
        products = abs_returns.iloc[:-1].values * abs_returns.iloc[1:].values
        bpv = Decimal(str((np.pi / 2) * math.fsum(products)))
        logging.info(f"Bi-power variation using non-zero returns: {bpv}")
        
        # Calculate truncation level
        v_n = self.alpha * (bpv * Decimal(str(self.estimator_weight))).sqrt() * dt ** self.gamma
        
        # Add a sanity check to prevent extremely small values
        #min_threshold = 0.5 * np.percentile(np.abs(log_returns[log_returns != 0]), 95)
        #v_n = max(v_n, min_threshold)
        
        return v_n

    def estimate_noise_variance(self, log_returns: np.ndarray) -> Decimal:
        """
        This method uses the assumption the the noise variance is constant
        """
        #non_zero_returns = log_returns[log_returns != 0]

        squared_sum = Decimal(0)
        for ret in log_returns:
            squared_sum += Decimal(str(ret)) ** 2
        return squared_sum / (Decimal(2) * Decimal(len(log_returns)))

    def get_weighted_avg_increases_vect(self, log_returns: np.ndarray) -> np.ndarray:
        # Create a view of the original array with overlapping windows
        # This uses numpy's stride_tricks to avoid copying data
        windows = sliding_window_view(log_returns, self.pre_averaging_window - 1)
        
        # Apply weights to each window using matrix multiplication
        avg_inc = scipy.linalg.blas.dgemv(1.0, windows, self.averaging_weights)
        
        return avg_inc[:-1] # to match the number of debiasing terms


    def get_debiasing_term_vect(self, log_returns: np.ndarray) -> np.ndarray:
        # Create a view of the original array with overlapping windows
        windows = sliding_window_view(log_returns, self.pre_averaging_window)
        
        # Square the returns and then apply weights
        squared_windows = windows ** 2
        debiasing_terms = scipy.linalg.blas.dgemv(1.0, squared_windows, self.debiasing_weights)
        
        return debiasing_terms

    def estimate_spot_vol(self, values: np.ndarray) -> np.ndarray:
        n_used_values = len(values)
        spot_volatilities = np.zeros(n_used_values, dtype=np.float64)
        
        
        for tau in range(n_used_values):
            # Calculate weight slice index 
            # To check correctenss note that when tau = 0 => weight_index = self.kernel_zero_index as intended
            weight_index = self.kernel_zero_index - tau
            
            # Get slice of kernel weights
            kernel_weights = self.kernel_values[weight_index:weight_index + n_used_values]
            
            # Normalize weights
            avg_weight = np.mean(kernel_weights)
            weights = kernel_weights / avg_weight
            
            # Use np.dot instead of element-wise multiplication and sum
            # This handles NaN values differently than nansum, so we need to mask them
            spot_vol = scipy.linalg.blas.ddot(weights, values)
            spot_volatilities[tau] = spot_vol / self.estimator_weight
        
        return spot_volatilities


    def estimate_vol_vol(self, spot_volatilities: np.ndarray, p) -> Decimal:
        """
        Estimates integrated volatility of volatility using sparse sampling
        
        Parameters:
        spot_volatilities : Series of spot volatility estimates
        p : Spacing parameter for sparse sampling
        """
        n = len(spot_volatilities)
        
        # Calculate squared differences of sparsely sampled spot volatilities
        ivv = Decimal(0)
        for i in range(0, n // p - 1):
            diff = Decimal(str(spot_volatilities[(i+1)*p])) - Decimal(str(spot_volatilities[i*p]))
            ivv += diff ** Decimal(2)
        
        return ivv

    def optimal_theta(self, spot_volatilities: np.ndarray, daily_vol: Decimal, dt: Decimal) -> Decimal:
        """
        This method implements equation (3.1)
        Assumes constant noise variance (here called gamma, not to be confused with the parameter self.gamma)
        """
        # Integrate squared spot volatility (c_t^2)
        squared_vols = spot_volatilities.astype(np.float64)**2
        squared_vols_list = squared_vols.tolist()
        integrated_vol_squared = Decimal(str(math.fsum(squared_vols_list))) * dt
        logging.info(f"Integrated vol squared: {integrated_vol_squared}")
        
        # Under constant noise variance assumption
        gamma = self.noise_variance
        integrated_gamma_ct = gamma * daily_vol  # ∫γ·c_t dt = γ·∫c_t dt
        logging.info(f"Integrated gamma * vol: {integrated_gamma_ct}")

        integrated_gamma_squared = gamma ** 2      # ∫γ^2 dt = γ^2·1

        logging.info(f"Integrated gamma squared: {integrated_gamma_squared}")
        
        # Calculate optimal theta squared
        term1 = self.Phi_12**2 * integrated_gamma_ct**2
        term2 = Decimal(3) * self.Phi_11 * self.Phi_22 * integrated_gamma_squared * integrated_vol_squared
        sqrt_term = (term1 + term2).sqrt()

        numerator = sqrt_term - self.Phi_12 * integrated_gamma_ct
        denominator = Decimal(3) * self.Phi_11 * integrated_gamma_squared
        
        theta_squared = numerator / denominator
        return (theta_squared).sqrt()

    def optimal_bandwidth(self, spot_volatilities: np.ndarray, daily_vol: Decimal, dt: Decimal, vol_vol: Decimal) -> Decimal:
        """
        Implements equation (3.3) from the paper to calculate optimal bandwidth
        """
        # Calculate necessary integrated quantities
        squared_vols = spot_volatilities.astype(np.float64)**2
        squared_vols_list = squared_vols.tolist()
        integrated_vol_squared = Decimal(str(math.fsum(squared_vols_list))) * dt
        
        # Under constant noise variance assumption
        gamma = self.noise_variance
        integrated_gamma_spot_vol = gamma * daily_vol
        integrated_gamma_squared = gamma**2
        
        # Calculate Theta(theta) from corollary (3.2)
        theta_term = (self.Phi_22/self.theta * integrated_vol_squared + 
                      Decimal(2) * self.Phi_12 * self.theta * integrated_gamma_spot_vol + 
                      self.Phi_11 * self.theta**3 * integrated_gamma_squared)
        
        # Calculate optimal bandwidth
        optimal_b = dt ** Decimal('0.25') * (Decimal(4) * theta_term  / vol_vol ).sqrt()
        
        return optimal_b

    def get_truncation_mask(self, avg_weighted_increases: np.ndarray) -> Tuple[np.ndarray, int]:
        truncation_mask = np.zeros_like(avg_weighted_increases, dtype=bool)

        # Element-wise comparison with Decimal precision
        for i in range(len(avg_weighted_increases)):
            value_decimal = Decimal(str(avg_weighted_increases[i]))
            truncation_mask[i] = abs(value_decimal) <= self.truncation_level

        truncated_count = sum(~truncation_mask)

        return truncation_mask, truncated_count
    
    def estimate_daily_vol(self, log_returns: np.ndarray, kernel_type: str, optimize_bandwidth = True, optimize_theta = False):
        allowed_kernel_types = {'non_truncated', 'truncated_1', 'truncated_2'}
        if kernel_type not in allowed_kernel_types:
            logging.error(f"Invalid kernel type: {kernel_type}. Allowed values are {allowed_kernel_types}")
            raise ValueError(f"Invalid kernel type: {kernel_type}. Allowed values are {allowed_kernel_types}")

        # The trading day is normalized to T=1
        log_returns = log_returns.astype(np.float64)
        n = Decimal(len(log_returns))
        dt = Decimal(1) / n

        self.init_parameters(log_returns)

        logging.info("Computing avg weigted increases and debiasing terms...")
        #avg_weighted_increases = log_returns.index.to_series().apply(lambda i: self.get_weighted_avg_inc(log_returns, i))
        #debiasing_terms = log_returns.index.to_series().apply(lambda i: self.get_debiasing_term(log_returns, i))
        avg_weighted_increases = self.get_weighted_avg_increases_vect(log_returns)
        debiasing_terms = self.get_debiasing_term_vect(log_returns)
        logging.info("Computed avg weighted increases and debiasing terms")

        truncation_mask, truncated_count = self.get_truncation_mask(avg_weighted_increases)
        total_count = len(truncation_mask)
        logging.info(f"Truncated {truncated_count}/{total_count} values ({truncated_count/total_count*100:.2f}%)")

        if kernel_type == "non_truncated":
            values = (avg_weighted_increases ** 2 - 0.5 * debiasing_terms)
        elif kernel_type == "truncated_1":
            # Only the squared pre-averaged returns are filtered by the truncation
            truncated_squared = np.where(
                truncation_mask,
                avg_weighted_increases ** 2,
                0
            )
            values = truncated_squared - 0.5 * debiasing_terms
            
        elif kernel_type == "truncated_2":
            # The entire term is filtered by the truncation
            values = np.where(
                truncation_mask,
                avg_weighted_increases ** 2 - 0.5 * debiasing_terms,
                0
            )

        logging.info("Computing spot volatilities...")
        # First iteration
        spot_volatilities_1 = self.estimate_spot_vol(values)
        daily_vol_1 = Decimal(str(math.fsum(spot_volatilities_1))) / n
        logging.info(f"Computed spot volatilities! First daily vol estimate: {daily_vol_1}")

        if not optimize_theta and not optimize_bandwidth:
            return daily_vol_1

        if optimize_theta:
            logging.info("Optimizing Theta...")
            new_theta = self.optimal_theta(spot_volatilities_1, daily_vol_1, dt)
            self.theta = new_theta
            logging.info(f"Theta optimized! New theta: {self.theta}")

            # Recalculate pre-averaging window with new theta
            self.pre_averaging_window = self.get_pre_averaging_window(dt)
            self.estimator_weight = self.get_estimator_weight()
            logging.info(f"New pre averaging window {self.pre_averaging_window}")
            self.estimator_weight = self.get_estimator_weight()

        if optimize_bandwidth:
            # The new optimal bandwidth is calculated after the new theta is calculated,
            # since the bandwidth optimization has the theta as a parameter, but not the
            # other way around
            
            logging.info("Optimizing bandwidth...")
            # A value of p of 300 would lead to sampling approximately every 5 minutes as done in their simulation
            vol_vol = self.estimate_vol_vol(spot_volatilities_1, p=300)
            logging.info(f"Volatility of volatility: {vol_vol}")
            self.bandwidth = self.optimal_bandwidth(spot_volatilities_1, daily_vol_1, dt, vol_vol)
            logging.info(f"Bandwidth optimized! New bandwidth: {self.bandwidth}")

        # These need to be recaulculated for any optimization
        self.kernel_values = self.compute_kernel_values(n)
        logging.info("Computing spot volatilities...")
        # Second iteration, which should be sufficient (typically values don't even change)
        avg_weighted_increases = self.get_weighted_avg_increases_vect(log_returns)
        debiasing_terms = self.get_debiasing_term_vect(log_returns)
        logging.info("Computed avg weighted increases and debiasing terms")

        truncation_mask, truncated_count = self.get_truncation_mask(avg_weighted_increases)
        total_count = len(truncation_mask)
        logging.info(f"Truncated {truncated_count}/{total_count} values ({truncated_count/total_count*100:.2f}%)")

        if kernel_type == "non_truncated":
            values = (avg_weighted_increases ** 2 - 0.5 * debiasing_terms)
        elif kernel_type == "truncated_1":
            # Only the squared pre-averaged returns are filtered by the truncation
            truncated_squared = np.where(
                truncation_mask,
                avg_weighted_increases ** 2,
                0
            )
            values = truncated_squared - 0.5 * debiasing_terms
            
        elif kernel_type == "truncated_2":
            # The entire term is filtered by the truncation
            values = np.where(
                truncation_mask,
                avg_weighted_increases ** 2 - 0.5 * debiasing_terms,
                0
            )

        spot_volatilities_2 = self.estimate_spot_vol(values)
        daily_vol_2 = Decimal(str(math.fsum(spot_volatilities_2))) / n
        logging.info(f"Computed spot volatilities! Second daily vol estimate: {daily_vol_2}")
        return daily_vol_2

