from .mrl_er import MatryoshkaERLoss
from .base_mrl import MatryoshkaContrastiveLoss
from .ese import ESELoss

criterion_dict = {
    "mrl": MatryoshkaContrastiveLoss,
    "ese": ESELoss,
    "mrl_er": MatryoshkaERLoss,  # Sử dụng cùng class nhưng sẽ có thêm phần effective rank penalty
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_dict.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_dict[args.kd_loss_type](args)