import sys
from contextlib import suppress
from typing import TypeVar, Any, AnyStr, Tuple
import inspect
from typing import NamedTuple


class AnnotatedMeta(NamedTuple):
    class_name: str
    args: tuple
    metadata: tuple


def get_annotation_module(annotation) -> str:
    # Special cases
    if annotation is None:
        return "builtins"

    if hasattr(annotation, "__module__"):
        return annotation.__module__

    if hasattr(annotation, "__origin__"):
        return annotation.__origin__.__module__

    raise ValueError("Cannot determine the module of {}".format(annotation))


def get_annotation_class_name(annotation, module: str) -> str:
    # Special cases
    if annotation is None:
        return "None"
    elif annotation is Any:
        return "Any"
    elif annotation is AnyStr:
        return "AnyStr"
    elif inspect.isfunction(annotation) and hasattr(annotation, "__supertype__"):
        return "NewType"

    if getattr(annotation, "__qualname__", None):
        return annotation.__qualname__
    elif getattr(
        annotation, "_name", None
    ):  # Required for generic aliases on Python 3.7+
        return annotation._name
    elif module in ("typing", "typing_extensions") and isinstance(
        getattr(annotation, "name", None), str
    ):
        # Required for at least Pattern and Match
        return annotation.name

    origin = getattr(annotation, "__origin__", None)
    if origin:
        if getattr(origin, "__qualname__", None):  # Required for Protocol subclasses
            return origin.__qualname__
        elif getattr(origin, "_name", None):  # Required for Union on Python 3.7+
            return origin._name
        else:
            return origin.__class__.__qualname__.lstrip(
                "_"
            )  # Required for Union on Python < 3.7

    annotation_cls = annotation if inspect.isclass(annotation) else annotation.__class__
    return annotation_cls.__qualname__.lstrip("_")


def get_annotation_args(annotation, module: str, class_name: str) -> Tuple:
    try:
        original = getattr(sys.modules[module], class_name)
    except (KeyError, AttributeError):
        ...
    else:
        if annotation is original:
            return ()  # This is the original, unparametrized type

    # Special cases
    if class_name in ("Pattern", "Match") and hasattr(
        annotation, "type_var"
    ):  # Python < 3.7
        return (annotation.type_var,)
    elif class_name == "ClassVar" and hasattr(
        annotation, "__type__"
    ):  # ClassVar on Python < 3.7
        return (annotation.__type__,)
    elif class_name == "NewType" and hasattr(annotation, "__supertype__"):
        return (annotation.__supertype__,)
    elif class_name == "Literal" and hasattr(annotation, "__values__"):
        return annotation.__values__
    elif class_name == "Generic":
        return annotation.__parameters__

    return getattr(annotation, "__args__", ())


def unpack_annotation(annotation) -> AnnotatedMeta:
    if annotation is None or annotation is type(None):  # noqa: E721
        return AnnotatedMeta(class_name="None", args=(), metadata=())

    # Type variables are also handled specially
    with suppress(TypeError):
        if isinstance(annotation, TypeVar) and annotation is not AnyStr:
            return AnnotatedMeta(
                class_name="TypeVar", args=(annotation.__name__,), metadata=()
            )

    module = get_annotation_module(annotation)
    class_name = get_annotation_class_name(annotation, module)
    args = get_annotation_args(annotation, module, class_name)
    return AnnotatedMeta(
        class_name=class_name,
        args=args,
        metadata=getattr(annotation, "__metadata__", ()),
    )
