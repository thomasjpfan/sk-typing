from typing import Optional
from typing import Union


from ._typing import Literal


from sklearn.isotonic import IsotonicRegression


class IsotonicRegressionAnnotation:
    __estimator__ = IsotonicRegression

    def __init__(
        self,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        increasing: Union[Literal["auto"], bool] = True,
        out_of_bounds: Literal["nan", "clip", "raise"] = "nan",
    ):
        pass
