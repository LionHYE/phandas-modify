# Quantitative Trading Backtesting Library

This project is a modified and extended version of [quantbai/phandas](https://github.com/quantbai/phandas).

The original project provides a foundation for quantitative trading research and backtesting.  
This version extends its functionality with additional tools for data acquisition, strategy evaluation, sensitivity analysis, and factor deployment.

## Features

### 1. Extended Data Acquisition

This library expands the available methods for acquiring market data, making it more flexible for different research and backtesting workflows.

Depending on the user's needs, data can be collected, processed, and prepared for strategy testing more conveniently.

### 2. Monte Carlo Backtesting

Monte Carlo backtesting has been added to help evaluate the robustness of trading strategies.

By repeatedly simulating different possible trading outcomes, users can better understand:

- Strategy stability
- Potential drawdown risk
- Distribution of returns
- Risk-adjusted performance
- Worst-case scenarios

This feature is useful for testing whether a strategy remains reliable under different market conditions.

### 3. 2D and 3D Sensitivity Analysis

This project includes both two-dimensional and three-dimensional sensitivity analysis tools.

These tools help users understand how strategy performance changes when key parameters are adjusted.

Supported analysis types include:

- 2D parameter sensitivity analysis
- 3D parameter sensitivity analysis
- Visualization of performance changes across parameter combinations

This is especially useful for:

- Parameter optimization
- Robustness testing
- Avoiding overfitting
- Understanding strategy behavior

### 4. Factor Deployment Helper

A helper function has been added for factor deployment.

When deploying a factor-based strategy, this function can return:

- Trading pairs
- Corresponding factor values
- Weight ratios based on factor values

This makes it easier to transform research results into deployable trading signals or portfolio weights.

## Project Origin

This project is based on:

[quantbai/phandas](https://github.com/quantbai/phandas)

Special thanks to the original author for providing the foundation of this library.

## Main Modifications

Compared with the original version, this modified version includes the following major changes:

1. Expanded data acquisition methods
2. Added Monte Carlo backtesting
3. Added 2D and 3D sensitivity analysis
4. Added a deployment helper function for returning trading pairs, factor values, and weight ratios

## Use Cases

This library is suitable for:

- Quantitative trading research
- Strategy backtesting
- Factor testing
- Parameter sensitivity analysis
- Monte Carlo simulation
- Portfolio weight generation
- Strategy deployment preparation

## Disclaimer

This project is intended for research and educational purposes only.

Trading financial markets involves risk.  
Past performance does not guarantee future results.  
Please use this library responsibly and perform your own risk assessment before applying any strategy in live trading.

## License

Please refer to the license of the original project and update this section according to your own modifications and distribution requirements.
