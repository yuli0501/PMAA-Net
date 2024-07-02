from torch import nn
from monai.losses import DiceCELoss

BCEWithLogitsLoss = nn.BCEWithLogitsLoss

class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dc_weight: float = 1.0,
    ):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.criterion = DiceCELoss(
            include_background=True,
            sigmoid=True,
            to_onehot_y=False,
            reduction="mean",
            lambda_dice=dc_weight,
            lambda_ce=ce_weight,
        )

    def forward(self, input, target):
        return self.criterion(input, target)
