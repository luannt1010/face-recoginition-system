from src.loss import AdaFaceLoss, ArcFaceLoss, BatchHardTripletLoss
from src.net import IResNetEncoder, SimpleResNet, MobileFaceNet

MODEL_MAP = {"iresnet": IResNetEncoder, "base": SimpleResNet, "mobile": MobileFaceNet}
LOSS_MAP = {"ada": AdaFaceLoss, "arc": ArcFaceLoss, "triplet": BatchHardTripletLoss}
