import numpy as np
from typing import Optional
from typing import Union
from typing import Callable

from ._typing import Literal
from ._typing import RandomStateType
from ._typing import ArrayLike
from ._typing import DType

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MultiLabelBinarizer


class BinarizerAnnotation:
    __estimator__ = Binarizer

    def __init__(self, threshold: float = 0.0, copy: bool = True):
        pass


class FunctionTransformerAnnotation:
    __estimator__ = FunctionTransformer

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
        pass


class KBinsDiscretizerAnnotation:
    __estimator__ = KBinsDiscretizer

    def __init__(
        self,
        n_bins: Union[int, ArrayLike] = 5,
        encode: Literal["onehot", "onehot-dense", "ordinal"] = "onehot",
        strategy: Literal["Uniform", "quantile", "kmeans"] = "quantile",
        dtype: Optional[DType] = None,
    ):
        pass


class KernelCentererAnnotation:
    __estimator__ = KernelCenterer

    def __init__(
        self,
    ):
        pass


class LabelBinarizerAnnotation:
    __estimator__ = LabelBinarizer

    def __init__(
        self, neg_label: int = 0, pos_label: int = 1, sparse_output: bool = False
    ):
        pass


class LabelEncoderAnnotation:
    __estimator__ = LabelEncoder

    def __init__(
        self,
    ):
        pass


class MaxAbsScalerAnnotation:
    __estimator__ = MaxAbsScaler

    def __init__(self, copy: bool = True):
        pass


class MinMaxScalerAnnotation:
    __estimator__ = MinMaxScaler

    def __init__(
        self, feature_range: tuple = (0, 1), copy: bool = True, clip: bool = False
    ):
        pass


class MultiLabelBinarizerAnnotation:
    __estimator__ = MultiLabelBinarizer

    def __init__(
        self, classes: Optional[ArrayLike] = None, sparse_output: bool = False
    ):
        pass


class NormalizerAnnotation:
    __estimator__ = Normalizer

    def __init__(self, norm: Literal["l1", "l2", "max"] = "l2", copy: bool = True):
        pass


class OneHotEncoderAnnotation:
    __estimator__ = OneHotEncoder

    def __init__(
        self,
        categories: Union[Literal["auto"], ArrayLike] = "auto",
        drop: Union[Literal["first", "if_binary"], ArrayLike, None] = None,
        sparse: bool = True,
        dtype: DType = np.float64,
        handle_unknown: Literal["error", "ignore"] = "error",
    ):
        pass


class OrdinalEncoderAnnotation:
    __estimator__ = OrdinalEncoder

    def __init__(
        self,
        categories: Union[Literal["auto"], ArrayLike] = "auto",
        dtype: DType = np.float64,
        handle_unknown: Literal["error", "ignore"] = "error",
        unknown_value: Union[int, float, None] = None,
    ):
        pass


class PolynomialFeaturesAnnotation:
    __estimator__ = PolynomialFeatures

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: Literal["C", "F"] = "C",
    ):
        pass


class PowerTransformerAnnotation:
    __estimator__ = PowerTransformer

    def __init__(
        self,
        method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
        standardize: bool = True,
        copy: bool = True,
    ):
        pass


class QuantileTransformerAnnotation:
    __estimator__ = QuantileTransformer

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = 100000,
        random_state: RandomStateType = None,
        copy: bool = True,
    ):
        pass


class RobustScalerAnnotation:
    __estimator__ = RobustScaler

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ):
        pass


class StandardScalerAnnotation:
    __estimator__ = StandardScaler

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ):
        pass
