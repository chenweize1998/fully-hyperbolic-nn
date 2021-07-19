"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.ltransformer import LorentzTransformerDecoder


str2dec = {"ltransformer": LorentzTransformerDecoder}

__all__ = ["DecoderBase",
           "str2dec",
           "LorentzTransformerDecoder"]
