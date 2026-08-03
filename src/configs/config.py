from src.loss import AdaFaceLoss, ArcFaceLoss, BatchHardTripletLoss
from src.net import IResNetEncoder, SimpleResNet

MODEL_MAP = {"iresnet": IResNetEncoder, "base": SimpleResNet}
LOSS_MAP = {"ada": AdaFaceLoss, "arc": ArcFaceLoss, "triplet": BatchHardTripletLoss}
