# Liquid Time-Constant Networks for Volatility Forecasting

This repository contains the implementation from my
Bachelor's thesis: *"Liquid Time-Constant Neural
Networks: A Continuous-Time Approach to Volatility
Forecasting"* (University of St. Gallen, 2025).
See the *Thesis.pdf* file for the full text.

## Abstract

Addressing the limitations of discrete-time models
in handling irregular financial data, this
thesis explores Liquid Time-Constant Networks
(LTCs) – biologically-inspired neural networks
operating in continuous time via ordinary
differential equations – for daily volatility
forecasting. This work evaluates LTCs against the
RealGARCH model using a kernel-estimated
realized volatility measure derived from
high-frequency IBM stock data (2000-2024). The two
models are evaluated on their one-step-ahead
forecasts throughout the COVID-19 crisis and the
following years until 2024. LTCs, especially when
incorporating logarithmically-transformed time
intervals between observations, consistently
outperform RealGARCH, achieving exceptionally
high statistical significance outside the crisis
period. Trained in only a few minutes on a single
CPU core, the LTCs demonstrated exceptional
computational and data efficiency while hinting
at a non-linear relationship between market
closure duration and volatility dynamics.

## 🚀 Quick Start

This repository uses the
[Nix](https://nixos.org) package
manager to install the necessary packages and make
them available in a reproducible shell. This
ensures that you use the exact same python and R
packages and package versions that I did in order
to make the results more easily reproducible and
generally avoid much of the struggle with mixing
packages from different languages. For this you
will need to [install
Nix](https://nixos.org/download/). It is available
on Linux, macOS, and Windows Subsystem for Linux
(WSL).

It is crucial that if you try to reproduce these
results you don't try to install the packages
yourself, for I used my own fork of the ncps
package, which implements the LTCs. This is due to
a bug in the original package, which is what you
would install using pip or something similar. The
Nix setup takes care of that.

### Setup

After installing Nix, run the following code:

```bash
git clone https://github.com/vxld100/ltc-volatility-forecasting.git
cd ltc-volatility-forecasting
nix develop
```

This last command will take a while. Please be
patient while Nix sets up the development
environment. If you plan to return to work with
this repository consider installing
[direnv](https://direnv.net/) which will cache the
shell and automatically enable it when you enter
the directory (run `direnv allow` after setup).
Don't forget to [enable
direnv](https://direnv.net/docs/hook.html) in your
shell of choice. Otherwise you might have to
download some or all of the packages the next time
you visit the directory.

### Key Files
- `data.py` - TAQ data processing and kernel volatility estimation
- `kernel.py` - my implementation of Figueroa-López & Wu (2024)
- `train_realgarch.R` - RealGARCH (Hansen et. al, 2012) model training
- `train_ltc.py` - LTC (Hasani et. al, 2021) training and comparison

## 🛠️ Usage

### Prerequisites
The code assumes the following directory structure:
- Create `data/raw/` directory 
- Place unprocessed TAQ trade files there
- Rename files to follow `ibm_*` pattern (currently hardcoded for IBM data)

### Step-by-step workflow

#### 1. Process TAQ data and calculate realized volatility
```bash
python3 data.py
```
This will:
- Clean the data with a sliding window using mean
  average deviation
- Calculate kernel-based realized volatility
- Output results to `data/output/ibm_realized_vol_prc_2000-2024.csv`

*Note: You never need to call `kernel.py` directly - it's kept separate for clarity.*

#### 2. Train RealGARCH benchmark
```bash
Rscript train_realgarch.R
```
This trains the RealGARCH model and saves predictions to the `data/` directory.

#### 3. Train LTC model and compare
```bash
python3 train_ltc.py
```
To change hyperparameters modify the dictionary at
the top of the main code execution section of the
file. The results are saved to `results/`
directory (not `result/` - that's for Nix builds).

### Understanding the output

Each training and evaluation run creates a
subdirectory like `00033928277844097-e7a648dbc4/`
where:
- **First part**: Scaled validation loss (lower = better performance)  
- **Second part**: MD5 hash of hyperparameters (same hash = same config)

Inside this directory you will find the following
files:

- **`.json`** - Complete hyperparameters and results (training time, metrics, etc.)
- **`.csv`** - Model comparison data (ground truth, predictions, log returns)
- **`.ckpt`** - Best model checkpoint (lowest validation loss)
- **`model_comparison.pdf`** - Visual comparison of LTC vs RealGARCH predictions
- **`training_loss.pdf`** - Training and validation loss curves

## 📝 Citation

If you use this code or the findings in your research, please cite:

```bibtex
@thesis{boschung2025ltc,
  title={Liquid Time-Constant Neural Networks: A Continuous-Time Approach to Volatility Forecasting},
  author={Boschung, Luca},
  year={2025},
  school={University of St. Gallen},
  type={Bachelor's Thesis}
}
```

## 🔗 References

- **Figueroa-López, J. E., & Wu, B. (2024)**. Kernel estimation of spot volatility with microstructure noise using pre-averaging. Econometric Theory, 40(3), 558– 607.
- **Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021, May)**. Liquid time-constant networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657-7666.
- **Hansen, P. R., Huang, Z., & Shek, H. H. (2012)**. Realized garch: a joint model for returns and realized measures of volatility. Journal of Applied Econometrics, 27(6), 877–906.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Luca Boschung**  
University of St. Gallen  
luca.boschung@student.unisg.ch
