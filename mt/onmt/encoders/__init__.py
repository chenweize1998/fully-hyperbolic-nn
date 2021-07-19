"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.ltransformer import LorentzTransformerEncoder


str2enc = {"ltransformer": LorentzTransformerEncoder}

__all__ = ["EncoderBase", "str2enc",
           "LorentzTransformerEncoder"]
