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

kernels_dict = {
    'RBF': gpytorch.kernels.RBFKernel,
    'Matern': gpytorch.kernels.MaternKernel,
    'Periodic': gpytorch.kernels.PeriodicKernel,
    'Linear': gpytorch.kernels.LinearKernel,
    'Polynomial': gpytorch.kernels.PolynomialKernel,
    'Cosine': gpytorch.kernels.CosineKernel,
}

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    global likelihood, model
    data = flask.request.get_json(force=True)
    print(data)
    x_train = torch.tensor(data['x_train'])
    y_train = torch.tensor(data['y_train'])
    x_test = torch.tensor(data['x_test'])
    kernel_name = data['kernel_name']
    kernel_params = data['kernel_params']
    print(kernel_name, kernel_params, type(kernel_params))
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = kernels_dict[kernel_name]()
    model = models.ExactGPModel(x_train, y_train, likelihood, kernel)

    hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1.),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
    'covar_module.outputscale': torch.tensor(2.),
    }

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

if __name__ == '__main__':
    app.run(debug=True)
