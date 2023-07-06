import flask
import torch
import gpytorch
import models
import numpy as np
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST"]}})


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = None
x_train = None

kernels_dict = {
    'RBF': gpytorch.kernels.RBFKernel,
    'Matern': gpytorch.kernels.MaternKernel,
    'Periodic': gpytorch.kernels.PeriodicKernel,
    'Linear': gpytorch.kernels.LinearKernel,
    'Polynomial': gpytorch.kernels.PolynomialKernel,
    'Cosine': gpytorch.kernels.CosineKernel,
}

kernel_params_keys = {
    'RBF': ['lengthscale'],
    'Matern': ['lengthscale', 'nu'],
    'Periodic': ['lengthscale', 'period_length'],
    'Linear': ['variance'],
    'Polynomial': ['power'],
    'Cosine': ['period_length'],
}

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    global likelihood, model, x_train
    data = flask.request.get_json(force=True)
    print(data)
    x_train = torch.tensor(data['x_train'])
    y_train = torch.tensor(data['y_train'])
    x_test = torch.tensor(data['x_test'])
    sigma = torch.tensor(data['sigma'])
    kernel_name = data['kernel_name']
    kernel_params = data['kernel_params']
    print(kernel_name, kernel_params, type(kernel_params))
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if kernel_name not in kernels_dict:
        return flask.jsonify({
            'mu': [],
            'lower': [],
            'upper': [],
        })
    elif kernel_name == 'Polynomial':
        kernel = kernels_dict[kernel_name](power=kernel_params[1])
    else:
        kernel = kernels_dict[kernel_name]()

    model = models.ExactGPModel(x_train, y_train, likelihood, kernel)

    hypers = {
    'likelihood.noise_covar.noise': sigma,
    'covar_module.outputscale': kernel_params[0]
    }
    for i, key in enumerate(kernel_params_keys[kernel_name]):
            hypers['covar_module.base_kernel.' + key] = kernel_params[i+1]

    model.initialize(**hypers)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
        lower, upper = observed_pred.confidence_region()
        mu = observed_pred.mean.numpy()
    
    return flask.jsonify({
        'mu': mu.tolist(),
        'lower': lower.numpy().tolist(),
        'upper': upper.numpy().tolist(),
    })

@app.route('/api/kernel', methods=['GET'])
@cross_origin()
def kernel():
    global likelihood, model, x_train

    if model is None:
        return flask.jsonify({
            'kernel_name': '',
            'kernel_params': {},
            'covariance_matrix': [[]],
        })

    covariance_matrix = model.covar_module(x_train).evaluate()

    return flask.jsonify({
        'covariance_matrix': covariance_matrix.detach().numpy().tolist(),
    })



if __name__ == '__main__':
    app.run(debug=True)
