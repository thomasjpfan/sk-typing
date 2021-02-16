import numpy as np
from typing import Optional
from typing import Union
from typing import Callable

from .typing import Literal
from .typing import RandomStateType
from .typing import ArrayLike
from .typing import DType


class Binarizer:
    def __init__(self, threshold: float = 0.0, copy: bool = True):
        ...


class FunctionTransformer:
    def __init__(
        self,
        func: Optional[Callable] = None,
        inverse_func: Optional[Callable] = None,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = True,
        kw_args: Optional[dict] = None,
        inv_kw_args: Optional[dict] = None,
    ):
        ...


class KBinsDiscretizer:
    def __init__(
        self,
        n_bins: Union[int, ArrayLike] = 5,
        encode: Literal["onehot", "onehot-dense", "ordinal"] = "onehot",
        strategy: Literal["Uniform", "quantile", "kmeans"] = "quantile",
        dtype: Optional[DType] = None,
    ):
        ...


class KernelCenterer:
    def __init__(
        self,
    ):
        ...


class LabelBinarizer:
    def __init__(
        self, neg_label: int = 0, pos_label: int = 1, sparse_output: bool = False
    ):
        ...


class LabelEncoder:
    def __init__(
        self,
    ):
        ...


class MaxAbsScaler:
    def __init__(self, copy: bool = True):
        ...


class MinMaxScaler:
    def __init__(
        self, feature_range: tuple = (0, 1), copy: bool = True, clip: bool = False
    ):
        ...


class MultiLabelBinarizer:
    def __init__(
        self, classes: Optional[ArrayLike] = None, sparse_output: bool = False
    ):
        ...


class Normalizer:
    def __init__(self, norm: Literal["l1", "l2", "max"] = "l2", copy: bool = True):
        ...


class OneHotEncoder:
    def __init__(
        self,
        categories: Union[Literal["auto"], ArrayLike] = "auto",
        drop: Union[Literal["first", "if_binary"], ArrayLike, None] = None,
        sparse: bool = True,
        dtype: DType = np.float64,
        handle_unknown: Literal["error", "ignore"] = "error",
    ):
        ...


class OrdinalEncoder:
    def __init__(
        self,
        categories: Union[Literal["auto"], ArrayLike] = "auto",
        dtype: DType = np.float64,
        handle_unknown: Literal["error", "ignore"] = "error",
        unknown_value: Union[int, float, None] = None,
    ):
        ...


class PolynomialFeatures:
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: Literal["C", "F"] = "C",
    ):
        ...


class PowerTransformer:
    def __init__(
        self,
        method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
        standardize: bool = True,
        copy: bool = True,
    ):
        ...


class QuantileTransformer:
    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = 100000,
        random_state: RandomStateType = None,
        copy: bool = True,
    ):
        ...


class RobustScaler:
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ):
        ...


class StandardScaler:
    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ):
        ...
