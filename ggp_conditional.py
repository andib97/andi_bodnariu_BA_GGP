#the following code is adapted from https://github.com/GPflow/GPflow/tree/develop/gpflow/conditionals

from gpflow.utilities import Dispatcher
conditional_train =Dispatcher("conditional_train")

import tensorflow as tf

from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.utilities.ops import eye
from gpflow.config import default_jitter
from gpflow.conditionals.util import base_conditional, expand_independent_outputs


@conditional_train.register(object, InducingVariables, Kernel, object)
def _conditional_train(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """
    Single-output GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    Kmm = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]
    Kmn = Kuf(inducing_variable, kernel, Xnew)  # [M, N]
    Knn = kernel.diag_tr() #uses optimzied function to calculate the covariances
    fmean, fvar = base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )  # [N, R],  [R, N, N] or [N, R]
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)

