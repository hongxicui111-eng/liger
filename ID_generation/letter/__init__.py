"""
LETTER tokenizer modules.

Implements the RQ-VAE tokenizer from LETTER (LEveraging TokenizER for Generative Recommendation),
which uses Sinkhorn-balanced assignment, diversity loss, and collaborative filtering (CF) loss
to produce better semantic IDs for generative recommendation.

Reference: https://github.com/HonghuiBao2000/LETTER
"""

from .rqvae import RQVAELetter
from .rq import ResidualVectorQuantizer
from .vq import VectorQuantizer
