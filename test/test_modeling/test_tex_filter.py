import torch

from pytools.modeling.TexFilter import SeqModel
from pytools.config import Config


def test_build_tex_filter_model(config:Config):

    m = SeqModel(7, config.model_pdt.filter_net)
    x = torch.rand(10, 7, 3)
    y= m.forward(x)
    assert list(y.shape)==[10, 8, 3]