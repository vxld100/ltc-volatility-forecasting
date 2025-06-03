# Liquid Time-Constant Networks for Volatility Forecasting

This repository implements the methodology from my
Bachelor's thesis: *"Liquid Time-Constant Neural
Networks: A Continuous-Time Approach to Volatility
Forecasting"* (University of St. Gallen, 2025).

## Abstract

Addressing the limitations of discrete-time models
in handling irregular financial data, this
thesis explores Liquid Time-Constant Networks
(LTCs) – biologically- inspired neural networks
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
packages that I did in order to make the results
more easily reproducible and generally avoid much
of the struggle with mixing packages from
different languages. For this you will need to
[install Nix](https://nixos.org/download/). It is
available on Linux, macOS, and Windows Subsystem
for Linux (WSL).

It is crucial that if you try to reproduce these
results you don't try to install the packages
yourself, for I used my own fork of the ncps
package, which implements the LTCs. This is due to
a bug in the original package, which is what you
would install using pip or something similar.

### Setup

```bash
git clone https://github.com/vxld100/ltc-volatility-forecasting.git
cd ltc-volatility-forecasting
nix develop
```

### Key Files
- `data.py` - TAQ data processing and kernel volatility estimation
- `kernel.py` - Figueroa-López & Wu (2024) implementation  
- `train_realgarch.R` - RealGARCH model training
- `train_ltc.py` - LTC training and comparison

## 📝 Citation

If you use this code in your research, please cite:

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

- **Figueroa-López, J.E. & Wu, B. (2024)**. Kernel estimation of spot volatility with microstructure noise using pre-averaging. *Econometric Theory*, 40(3), 558-607.
- **Hasani, R. et al. (2021)**. Liquid time-constant networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666.
- **Hansen, P.R., Huang, Z. & Shek, H.H. (2012)**. Realized GARCH: A joint model for returns and realized measures of volatility. *Journal of Applied Econometrics*, 27(6), 877-906.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Luca Boschung**  
University of St. Gallen  
luca.boschung@student.unisg.ch
