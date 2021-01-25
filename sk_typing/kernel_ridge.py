from typing import Optional
from typing import Union
from typing import Callable


from sklearn.kernel_ridge import KernelRidge


class KernelRidgeAnnotation:
    __estimator__ = KernelRidge

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: Union[str, Callable] = "linear",
        gamma: Optional[float] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
    ):
        pass
