from typing import Optional
from typing import Union
from typing import Callable

from ._typing import RandomStateType

from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import SkewedChi2Sampler


class AdditiveChi2SamplerAnnotation:
    __estimator__ = AdditiveChi2Sampler

    def __init__(self, sample_steps: int = 2, sample_interval: Optional[float] = None):
        pass


class NystroemAnnotation:
    __estimator__ = Nystroem

    def __init__(
        self,
        kernel: Union[str, Callable] = "rbf",
        gamma: Optional[float] = None,
        coef0: Optional[float] = None,
        degree: Optional[float] = None,
        kernel_params: Optional[dict] = None,
        n_components: int = 100,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        pass


class PolynomialCountSketchAnnotation:
    __estimator__ = PolynomialCountSketch

    def __init__(
        self,
        gamma: float = 1.0,
        degree: int = 2,
        coef0: int = 0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        pass


class RBFSamplerAnnotation:
    __estimator__ = RBFSampler

    def __init__(
        self,
        gamma: float = 1.0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        pass


class SkewedChi2SamplerAnnotation:
    __estimator__ = SkewedChi2Sampler

    def __init__(
        self,
        skewedness: float = 1.0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        pass
