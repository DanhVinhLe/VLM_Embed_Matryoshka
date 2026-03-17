from .adaptive_inference import route_and_truncate_query
from .base_mrl import MatryoshkaContrastiveLoss
from .ese import ESELoss
from .adaptive_matryoshka import AdaptiveMatryoshkaStage1Loss, AdaptiveRouterLoss

criterion_dict = {
    "mrl": MatryoshkaContrastiveLoss,
    "ese": ESELoss,
    "adaptive_mrl_stage1": AdaptiveMatryoshkaStage1Loss,
    "adaptive_router": AdaptiveRouterLoss,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_dict.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_dict[args.kd_loss_type](args)