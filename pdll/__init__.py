"""Pairwise Difference Learning (PDL): scikit-learn meta-estimators that learn from pairs of points.

If you use pdll in your research, please cite the paper (``pdll.__citation__`` holds the BibTeX):
Belaid, Rabus & Hüllermeier, *Pairwise Difference Learning for Classification*, Discovery Science 2024.
"""
from ._classifier import PairwiseDifferenceClassifier
from ._regressor import PairwiseDifferenceRegressor
from ._transformer import PDCDataTransformer

__all__ = ['PairwiseDifferenceClassifier', 'PairwiseDifferenceRegressor', 'PDCDataTransformer']

__version__ = '1.0.0'

__copyright__ = 'Copyright 2024, Mohamed Karim Belaid'
__license__ = 'Apache License 2.0'

__citation__ = """@inproceedings{belaid2024pairwise,
  title     = {Pairwise Difference Learning for Classification},
  author    = {Belaid, Mohamed Karim and Rabus, Maximilian and H{\\"u}llermeier, Eyke},
  booktitle = {Discovery Science},
  pages     = {284--299},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-78980-9_18},
  url       = {https://arxiv.org/abs/2406.20031}
}"""
