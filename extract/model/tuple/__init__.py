from .cardinal import CardinalPredictor
from .money import MoneyPredictor
from .percent import PercentPredictor
from .quantity import QuantityPredictor

def get_predictor(label):
    if label == "CRD" or label == "CARDINAL":
        return CardinalPredictor
    elif label == "QTY" or label == "QUANTITY":
        return QuantityPredictor
    elif label == "PRC" or label == "PERCENT":
        return PercentPredictor
    elif label == "MON" or label == "MONEY":
        return MoneyPredictor