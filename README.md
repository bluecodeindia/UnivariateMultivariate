# Univariate and Multivariate Models for Tangni Landslide Prediction

## Overview
This repository contains various Jupyter notebooks for implementing and evaluating different univariate and multivariate machine learning models for predicting the Tangni landslide. The models include Autoregression, LSTM, MLP, SARIMA, SARIMAX, and SMO.

## Repository Structure
- `.ipynb_checkpoints/`: Jupyter notebook checkpoints.
- `Autoregression.ipynb`: Notebook for autoregression model.
- `LSTM Multivariate.ipynb`: Multivariate LSTM model for landslide prediction.
- `LSTM Univariate.ipynb`: Univariate LSTM model for landslide prediction.
- `MLP Multivariate.ipynb`: Multivariate MLP model for landslide prediction.
- `MLP Univariate.ipynb`: Univariate MLP model for landslide prediction.
- `SARIMA Univariate grid search.ipynb`: Grid search for SARIMA model parameters in univariate setting.
- `SARIMA Univariate.ipynb`: Univariate SARIMA model for landslide prediction.
- `SARIMAX with exogenous variable.ipynb`: SARIMAX model with exogenous variables for landslide prediction.
- `SMO Univariate.ipynb`: Univariate SMO model for landslide prediction.
- `Vectored Autoregression.ipynb`: Vector autoregression model.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Statsmodels
- Matplotlib

You can install them using pip:
```bash
pip install numpy pandas scikit-learn tensorflow keras statsmodels matplotlib
```

### Running the Notebooks
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/bluecodeindia/UnivariateMultivariate.git
    cd UnivariateMultivariate
    ```

2. **Open the Notebooks**:
    Use Jupyter Notebook or JupyterLab to open the notebooks. For example:
    ```bash
    jupyter notebook LSTM\ Multivariate.ipynb
    ```

3. **Run the Cells**:
    Execute the cells in each notebook sequentially to train and evaluate the models.

## Notebooks

### Univariate Models
- `Autoregression.ipynb`: Implements an autoregression model for univariate time series prediction.
- `LSTM Univariate.ipynb`: Implements a univariate LSTM model for landslide prediction.
- `MLP Univariate.ipynb`: Implements a univariate MLP model for landslide prediction.
- `SARIMA Univariate grid search.ipynb`: Performs grid search to find the best parameters for SARIMA model.
- `SARIMA Univariate.ipynb`: Implements a univariate SARIMA model for landslide prediction.
- `SMO Univariate.ipynb`: Implements a univariate SMO model for landslide prediction.

### Multivariate Models
- `LSTM Multivariate.ipynb`: Implements a multivariate LSTM model for landslide prediction.
- `MLP Multivariate.ipynb`: Implements a multivariate MLP model for landslide prediction.
- `SARIMAX with exogenous variable.ipynb`: Implements a SARIMAX model with exogenous variables for landslide prediction.
- `Vectored Autoregression.ipynb`: Implements a vector autoregression model for multivariate time series prediction.

## Contributing
Feel free to contribute by submitting a pull request. Please ensure your changes are well-documented and tested.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact [bluecodeindia@gmail.com](mailto:bluecodeindia@gmail.com).
