"""
Class file for a flexible Bayesian Optimisation model using the BOTorch library.
Can be used to implement single and multi objective Bayesian Optimisation
Author: Terrence A'Hearn
Last edited: 21/09/2024

For more information on the implementation and how the library works visit the BOTorch website:
https://botorch.org/tutorials/multi_objective_bo

"""

import torch
import warnings
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement


class BayesianModel():
    
    def __init__(self, domain, x_init, y_init) -> None:
        """
        This class maintains a Bayesian probabalistic model of a black box function using a Gaussian Process prior.
        It can be used to find the global MAXIMUM of a function.

        :param domain: A collection that specifies the minimum and maximum values of each input parameter.
                       It should be an (2 x d) pyTorch tensor as follows:
                            [
                            [X1_min, X2_min, ..., Xd_min],
                            [X1_max, X2_max, ..., Xd_max]
                            ]

                            Where:
                                * d is the dimension of the input space

        :param x_init: The initial set of observed points in the domain. It should be an (n x d) pyTorch tensor as follows:
                            [
                            [X1_1, X1_2, ..., X1_d],
                            [X2_1, X2_2, ..., X2_d],
                            ...,
                            [Xn_1, Xn_2, ..., Xn_d]
                            ]

                            Where:
                                * d is the dimension of the input space
                                * n is the number of observed points
                                * each Xi_j is the j-th parameter of the i-th point

        :param y_init: The initial set of observed values of the objective functions which are associated with the corresponding
                       input from x_obs. It should be an (n x m) pyTorch tensor as follows:
                            [
                            [Y1_1, Y1_2, ..., Y1_m],
                            [Y2_1, Y2_2, ..., Y2_m],
                            ...,
                            [Yn_1, Yn_2, ..., Yn_m]
                            ]
                            
                            Where:
                                * n is the number of observed points
                                * m is the number of objectives
                                * each Yi_j is the j-th objective value for the i-th observed point
                                
        """

        # Initial Data
        self.x_obs = x_init
        self.y_obs = y_init

        # Problem Domain
        self.domain = domain
        self.dimension = self.domain.shape[1]
        assert self.dimension >= 1, "No input parameters specified!"

        # Objectives
        self.objectives = self.y_obs.shape[-1]
        assert self.objectives >= 1, "No objective values specified!"

        # Model
        self.model = self._initialiseModel()
        self._fitModel()

    def predict(self, X, return_std=False):
        """
        A method for predicting the objective values at some given points, as well as the associated uncertainty.
        The returned estimate is the mean of the internal Gaussian Process evaluated at those points, and there
        is the option to return the standard deviation of the Gaussian at those points.

        :param X: The desired points to estimate the function at . It should be an (n x d) pyTorch tensor as follows:
                    [
                    [X1_1, X1_2, ..., X1_d],
                    [X2_1, X2_2, ..., X2_d],
                    ...,
                    [Xn_1, Xn_2, ..., Xn_d]
                    ]

                    Where:
                        * d is the dimension of the input space
                        * n is the number of observed points
                        * each Xi_j is the j-th parameter of the i-th point

        :param return_std: A boolean value specifying whether the standard deviation of the estimates should be returned.

        :returns: An (n x m) pyTorch tensor containing the predicted values of the functions at the given points, as well
                    as optionally the standard deviation of the estimate.

        """

        self.model.eval()

        with torch.no_grad():
            
            posterior = self.model.posterior(normalize(X, self.domain))

        mean = posterior.mean

        if return_std:

            std = posterior.mvn.stddev
            std = std.reshape(-1, self.objectives)

            return mean, std
        
        return mean
        

    def suggest(self, batch_size=1, boundaries=None, ref_point=None, noise=None, X_pending=None):
        """
        This method can be called to suggest a batch of points to be evaluated acording to the acquisition function. It should balance
        suggesting points with high unceratinty (exploration) with suggestion points in promising areas (exploitation).
        !! Important !! This implementation will suggest based on searching for the global MAXIMUM of the function. If a global minimum
        is required you should negate the objective.

        :param batch_size: The number of points to suggest at this iteration

        :param boundaries: The input boundaries in which the model should suggest points. If not specified then the boundaries are taken
                            to be the entire domain that the model was initialised with. Otherwise it should take the form of a
                            (2 x d) pyTorch tensor where d is the input dimension as follows:
                            [
                            [X1_min, X2_min, ..., Xd_min],
                            [X1_max, X2_max, ..., Xd_max]
                            ]

        :param ref_point: Only needed when using the model for multiple objectives. It is the lower bound of the objectives and should take
                            the form of a (1 x m) pyTorch tensor where m is the number of objectives as follws:
                            [R1, R2, ..., Rm]

        :param noise: Determines the acquisition function to be used for multi-objective optimisation. If True, then qlNEHVI is used, if False
                        then qlEHVI is used, if None then it is chosen based on batch size. qlNEHVI is more efficient for batch sizes > 1and so it
                        is True by default. See the BOTorch docs linked at the beginning of the module for more information on these functions.

        :param X_pending: An optional tensor of points that are pending evaluation so that they are not suggested again. It should take the form
                            of an (p x d) pyTorch tensor where p is the number of pending points and d is the dimension of the input space as follows:
                            [
                            [X1_1, X1_2, ..., X1_d],
                            [X2_1, X2_2, ..., X2_d],
                            ...,
                            [Xp_1, Xp_2, ..., Xp_d]
                            ]

        :returns: An (batch_size x d) pyTorch tensor of suggested points, where d is the dimension of the input space.

        """

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Inferring no pending points from an empty tensor in X_pending
        if X_pending is None or len(X_pending) == 0:
            X_pending = None

        # Inferring noisy acquisition from unspecified noise
        if noise is None and batch_size > 1:
            noise = True

        # Assigning the boundaries as the domain if none were specified
        if boundaries is None:
            boundaries = self.domain

        # Normalizes the boundaries to the domain scale, as suggestions are computed in normalized form
        boundaries = normalize(boundaries, self.domain)

        # Defining the sampler for optimization
        SAMPLES = 128
        normSampler = SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLES]))

        if self.objectives > 1:     # Suggesting points for multi-acquisition
            
            assert ref_point is not None, "Must include a reference point. See documentation for more information"

            # Defining acquisition function
            if not noise:

                # for qEHVI we must first partition the non dominated space into disjoint rectangles
                with torch.no_grad():
                    pred = self.model.posterior(normalize(self.x_obs, self.domain)).mean

                partitioning = FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=pred
                )

                acq_func = qLogExpectedHypervolumeImprovement(
                    model=self.model,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=normSampler,
                    X_pending=X_pending
                )

            else:
                acq_func = qLogNoisyExpectedHypervolumeImprovement(
                    model=self.model,
                    ref_point=ref_point,
                    X_baseline=normalize(self.x_obs, self.domain),
                    prune_baseline=True,
                    sampler=normSampler,
                    X_pending=X_pending
                )


        else:   # Suggesting points for single-acquisition

            # Define acquisition function
            max_f = self.y_obs.max().item()
            acq_func = qLogExpectedImprovement(
                model=self.model,
                best_f=max_f,
                sampler=normSampler,
                X_pending=X_pending
            )

        # Get candidates
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=boundaries,
            q=batch_size,
            num_restarts=10, # Hyper param
            raw_samples=512, # Hyper param
            options={"maxiter":200}, # Hyper param
            sequential=batch_size>1 # Sequential greedy optimization if batch size is greater than 1
        )

        return unnormalize(candidates, bounds=self.domain)

        
    def _initialiseModel(self):
        """
        Internal method for initialising the models for each objective

        """
        
        models = [
            SingleTaskGP(
                train_X=normalize(self.x_obs, self.domain),
                train_Y=self.y_obs[:,i].reshape(-1,1),
                input_transform=Normalize(d=self.dimension),
                outcome_transform=Standardize(m=1)
                # Cam add the parameter covar_module to specify a kernel. While omitted it will use MaternKernel
            )
            for i in range(self.objectives)
        ]

        model = ModelListGP(*models)

        return model
        

    def _fitModel(self) -> None:
        """
        Internal method for fitting the model. This is done when the object is initialised.
        
        """

        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)

        fit_gpytorch_mll(mll)
