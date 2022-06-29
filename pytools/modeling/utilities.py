from typing import Tuple, Union, Iterable

import numpy as np
from numpy.core._multiarray_umath import ndarray


def list_it_2(element):
    return [element, element]


def get_cnn1d_dim(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    Get the output dimension of 1D cnn
    Use the formula at https://pytorch.org/docs/stable/nn.html; round down if not integer

    Args:
        length_in:
        kernel_size:
        stride:
        padding: same as conv1d default
        dilation: same as conv1d default

    Returns: length_out

    """
    length_out = (
        length_in + 2 * padding - dilation * (kernel_size - 1) - 1
    ) / stride + 1
    return int(length_out)


def get_cnn2d_dim(h_in, w_in, kernel=(3, 4), stride=(1, 1), padding=0, dilation=1):
    """
    Get the output height and weight for a conv2d layer

    Args:
        h_in: height in
        w_in: width in
        kernel: 2d kernel for height and weight
        stride:
        padding: same as conv1d default
        dilation: same as conv1d default

    Returns: out height and weight

    """
    if isinstance(kernel, int):
        kernel = list_it_2(kernel)
    if isinstance(stride, int):
        stride = list_it_2(stride)
    h_out = get_cnn1d_dim(
        length_in=h_in,
        kernel_size=kernel[0],
        stride=stride[0],
        padding=get_cnn_padding_1d(kernel_size=kernel[0]),
    )
    w_out = get_cnn1d_dim(
        length_in=w_in,
        kernel_size=kernel[1],
        stride=stride[1],
        padding=get_cnn_padding_1d(kernel_size=kernel[1]),
    )
    return h_out, w_out


def get_cnn_padding(kernel_sizes: Union[Tuple[int], int]) -> Union[Tuple[int], int]:
    if isinstance(kernel_sizes, int):
        return get_cnn_padding_1d(kernel_sizes)
    return tuple(get_cnn_padding_1d(k) for k in kernel_sizes)


def get_cnn_padding_1d(kernel_size: int, odd_only=True) -> int:
    """
    Get the padding dimensions for a given kernel size.
    Kernel size shall be an odd number usually.

    Args:
        kernel_size: an integer for one dimension of the kernel
        odd_only:

    Returns:

    """
    if odd_only:
        assert kernel_size % 2 == 1, "odd kernel sizes only"
    return int((kernel_size - 1) / 2)


def load_npz_as_dict(
    npz_file, map_col=("weather", "lag_load", "calendar_data", "y_labels")
):
    data: Union[ndarray, Iterable, int, float, tuple, dict] = np.load(npz_file)
    if not map_col:
        return {k: data[k] for k in data.files}
    else:
        return {map_col[i]: data[k] for i, k in enumerate(data.files)}


def extract_a_field(obj, attribute: str, default_val):
    if hasattr(obj, attribute):
        return eval(f"obj.{attribute}")
    else:
        return default_val


def extract_model_settings(train_opt, model_fields):
    return {f: train_opt.__dict__[f] for f in train_opt.__dict__ if f in model_fields}
