from typing import Optional
from typing import Union
from typing import Callable
from typing import Mapping
from typing import Iterable

import numpy as np

from .typing import Literal
from .typing import RandomStateType
from .typing import DType


class CountVectorizer:
    def __init__(
        self,
        input: Literal["filename", "file", "content"] = "content",
        encoding: str = "utf-8",
        decode_error: Literal["strict", "ignore", "replace"] = "strict",
        strip_accents: Optional[Literal["ascii", "unicode"]] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        stop_words: Union[Literal["english"], Callable, None] = None,
        token_pattern: str = "(?u)\\b\\w\\w+\\b",
        ngram_range: tuple = (1, 1),
        analyzer: Union[Literal["word", "char", "char_wb"], Callable] = "word",
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
        vocabulary: Union[Mapping, Iterable, None] = None,
        binary: bool = False,
        dtype: DType = np.int64,
    ):
        ...


class DictVectorizer:
    def __init__(
        self,
        dtype: DType = np.float64,
        separator: str = "=",
        sparse: bool = True,
        sort: bool = True,
    ):
        ...


class FeatureHasher:
    def __init__(
        self,
        n_features: int = 1048576,
        input_type: Literal["dict", "pair", "string"] = "dict",
        dtype: DType = np.float64,
        alternate_sign: bool = True,
    ):
        ...


class HashingVectorizer:
    def __init__(
        self,
        input: Literal["filename", "file", "content"] = "content",
        encoding: str = "utf-8",
        decode_error: Literal["strict", "ignore", "replace"] = "strict",
        strip_accents: Optional[Literal["ascii", "unicode"]] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        stop_words: Union[Literal["english"], list, None] = None,
        token_pattern: str = "(?u)\\b\\w\\w+\\b",
        ngram_range: tuple = (1, 1),
        analyzer: Union[Literal["word", "char", "char_wb"], Callable] = "word",
        n_features: int = 1048576,
        binary: bool = False,
        norm: Literal["l1", "l2"] = "l2",
        alternate_sign: bool = True,
        dtype: DType = np.float64,
    ):
        ...


class PatchExtractor:
    def __init__(
        self,
        patch_size: tuple = None,
        max_patches: Union[int, float, None] = None,
        random_state: RandomStateType = None,
    ):
        ...


class TfidfTransformer:
    def __init__(
        self,
        norm: Literal["l1", "l2"] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ):
        ...


class TfidfVectorizer:
    def __init__(
        self,
        input: Literal["filename", "file", "content"] = "content",
        encoding: str = "utf-8",
        decode_error: Literal["strict", "ignore", "replace"] = "strict",
        strip_accents: Literal["ascii", "unicode"] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        analyzer: Union[Literal["word", "char", "char_wb"], Callable] = "word",
        stop_words: Union[Literal["english"], list, None] = None,
        token_pattern: str = "(?u)\\b\\w\\w+\\b",
        ngram_range: tuple = (1, 1),
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
        vocabulary: Union[Mapping, Iterable, None] = None,
        binary: bool = False,
        dtype: DType = np.float64,
        norm: Literal["l1", "l2"] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ):
        ...
