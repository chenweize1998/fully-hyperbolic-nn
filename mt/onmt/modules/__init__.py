"""  Attention and normalization modules  """
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute

import onmt.modules.source_noise  # noqa

__all__ = ["CopyGenerator", 
           "CopyGeneratorLoss", "CopyGeneratorLossCompute"]
