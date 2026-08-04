from src.loss import AdaFaceLoss, ArcFaceLoss, BatchHardTripletLoss
from src.net import IResNetEncoder, SimpleResNet, mobilenet_encoder

MODEL_MAP = {"iresnet": IResNetEncoder, "base": SimpleResNet, "mobile": mobilenet_encoder}
LOSS_MAP = {"ada": AdaFaceLoss, "arc": ArcFaceLoss, "triplet": BatchHardTripletLoss}
