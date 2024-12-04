import torch

from pytools.modeling.TexFilter import Model
from pytools.config import Config


def test_build_tex_filter_model(config:Config):

    m = Model(config.model_pdt.filter_net)
    x = torch.rand(10, 7, 3)
    m.forward(x)