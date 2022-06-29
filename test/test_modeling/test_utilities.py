import pytest

from pytools.modeling.utilities import (
    get_cnn1d_dim,
    get_cnn2d_dim,
    get_cnn_padding,
    get_cnn_padding_1d,
    extract_a_field,
)


@pytest.mark.parametrize("stride, expected", [(1, 10), (2, 5), (3, 4)])
def test_get_cnn1d_dim(stride, expected):
    len_in = 10
    kernel = 5
    len_out = get_cnn1d_dim(
        length_in=len_in,
        kernel_size=kernel,
        stride=stride,
        padding=get_cnn_padding(kernel_size=kernel),
    )
    assert len_out == expected


@pytest.mark.parametrize(
    "stride, expected", [((1, 1), (10, 8)), ((2, 2), (5, 4)), ((3, 3), (4, 3))]
)
def test_get_cnn2d_dim(stride, expected):
    kernel = (5, 3)
    h_in = 10
    w_in = 8
    padding_h = get_cnn_padding(kernel_size=kernel[0])
    padding_w = get_cnn_padding(kernel_size=kernel[1])
    padding = [padding_h, padding_w]
    assert (
        get_cnn2d_dim(
            w_in=w_in, h_in=h_in, kernel=kernel, stride=stride, padding=padding
        )
        == expected
    )


@pytest.mark.parametrize("kernel, expected", [(5, 2), (3, 1), (7, 3)])
def test_get_cnn_padding_1d(kernel, expected):
    assert get_cnn_padding_1d(kernel_size=kernel) == expected


@pytest.mark.parametrize("kernels, expected", [((5, 3, 7), (2, 1, 3)), (5, 2)])
def test_get_cnn_padding(kernels, expected):
    assert get_cnn_padding(kernel_sizes=kernels)


def test_extract_a_field():
    class O:
        def __init__(self):
            self.x = "abc"

    o = O()
    assert extract_a_field(o, "x", default_val="default") == "abc"
