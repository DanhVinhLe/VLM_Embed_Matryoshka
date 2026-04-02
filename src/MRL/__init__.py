from .eofd import EOFDLoss
from .base_mrl import MatryoshkaContrastiveLoss
from .ese import ESELoss

criterion_dict = {
    "mrl": MatryoshkaContrastiveLoss,
    "ese": ESELoss,
    "eofd": EOFDLoss,  # Sử dụng cùng class nhưng sẽ có thêm phần effective rank penalty
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_dict.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_dict[args.kd_loss_type](args)