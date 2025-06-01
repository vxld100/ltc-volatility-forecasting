suppressMessages(library(rugarch))
suppressMessages(library(xts))
suppressMessages(library(zoo))

data <- read.csv("./data/output/ibm_realized_vol_prc_2000-2024.csv")
data$date <- as.Date(data$DATE, format="%Y-%m-%d")

# Calculate returns
returns <- diff(log(data$close_price))
real_vol <- data$daily_volatility[2:length(data$daily_volatility)]

# Create xts objects using the existing date column
dates <- data$date[2:length(data$date)]  # Skip first date since returns lose one observation
returns_xts <- xts(returns, order.by = dates)
real_vol_xts <- xts(real_vol, order.by = dates)

n_test = as.integer(0.2*length(returns))
test_data <- tail(real_vol_xts, n_test) # Keep test data aside for final evaluation
# Define test sample size

# Fit model with out.sample parameter to reserve data for testing
spec <- ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "sstd"
)

fit <- ugarchfit(
  spec = spec, 
  data = returns_xts, 
  realizedVol = real_vol_xts,
  out.sample = n_test
)

# Now forecast - n.roll must be less than or equal to out.sample
forc <- ugarchforecast(
  fit, 
  n.ahead = 1,
  n.roll = n_test - 1,  # Use one less than n_test to stay within constraints
  realizedVol = tail(real_vol_xts, n_test)  # Only need the realized vol for the test period
)

# Extract forecasted volatility
forecast_sigma <- sigma(forc)
forecast_variance <- forecast_sigma^2

n_forecasts <- length(as.vector(sigma(forc)))
# Extract the appropriate number of actual values
actual_realized_vol <- tail(real_vol_xts, length(forecast_variance))
#print(actual_realized_vol2)

# Calculate performance metrics for volatility forecasting
mse_vol <- mean((as.vector(actual_realized_vol) - as.vector(forecast_variance))^2)
print(mse_vol)

test_dates <- index(test_data)
plot_df <- data.frame(
  Date = test_dates,
  Actual_Variance = as.vector(actual_realized_vol),      # Use the simple vector
  Predicted_Variance = as.vector(forecast_variance) # Use the simple vector
)

save_df <- data.frame(
  DATE = test_dates,
  realGARCH_predictions = as.vector(forecast_variance)
)

write.csv(save_df, file="./data/rgarch_test_preds_v2.csv", row.names=FALSE)
