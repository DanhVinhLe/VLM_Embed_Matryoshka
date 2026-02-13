import json
from prepare_one_model_training import TrainOneModelCollator, OneModelTrainer, TrainOneModelDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.MRL import build_criterion
import time 
import os
import sys
from tqdm import tqdm 
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
import logging

# Set the logging level for Numba's CUDA driver
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)

# You may also want to set the general Numba logger level
logging.getLogger('numba').setLevel(logging.WARNING)
# Todo
# wandb.login(key="f5a118efa8813fb4edc7f6b8a7ab5c9c5f9e1ece")

def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    return optimizer

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, trainer, train_data, optimizer, lr_scheduler, criterion, model_args, training_args):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        # self.gpu_id = 0
        # self.gpu_id = int(training_args.gpu_id)
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.trainer = trainer.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args
        
        self.trainer = DDP(self.trainer, 
                             device_ids=[self.gpu_id],
                             find_unused_parameters=True)
    
    def _debug_batch_devices(self, obj, prefix=""):
        if obj is None:
            print(f"{prefix}Value: None")
            return
        
        try:
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    print(f"{prefix}Empty dict")
                for k, v in obj.items():
                    self._debug_batch_devices(v, prefix=f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    print(f"{prefix}Empty {type(obj).__name__}")
                for i, v in enumerate(obj):
                    self._debug_batch_devices(v, prefix=f"{prefix}[{i}].")
            else:
                print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
        except Exception as e:
            print(f"{prefix}ERROR: {e}")
        
    def run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        total_losses, contrastive_losses = [], []
        
        progress_bar = tqdm(total=len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps // dist.get_world_size(), 
                            desc=f"Epoch {epoch}",
                            disable=not dist.get_rank() == 0)
        for batch_idx, batch in enumerate(self.train_data):
            batch = to_device(batch, self.device)
            # with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            loss_dict = self.trainer(self.criterion, batch)

            total_loss = loss_dict['loss'] / self.training_args.gradient_accumulation_steps
            contrastive_loss = loss_dict['contrastive_loss']

            total_losses.append(loss_dict['loss'].detach().item())
            contrastive_losses.append(contrastive_loss.detach().item())
            
            batch_loss = sum(total_losses)/len(total_losses)
            batch_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
            
            total_loss.backward()
            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                if is_main_process():
                    progress_bar.set_postfix({
                        "loss": f"{batch_loss:.4f}",
                        "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                        "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                    progress_bar.update(1)
            torch.cuda.empty_cache()
        progress_bar.close()
        
    def train(self):
        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if is_main_process() and self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-epoch-{epoch}")
                projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                student = self.trainer.module.model
                student.encoder.save_pretrained(ckpt_dir)
                if self.model_args.model_backbone in ["llava_onevision", "llava_two_vision"]:
                    torch.save(student.encoder.model.multi_modal_projector.state_dict(), projector_dir)
                elif self.model_args.model_backbone in ["llava_qwen2"]:
                    torch.save(student.encoder.model.model.mm_projector.state_dict(), projector_dir)
                    
                student_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if student_config:
                    student_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                print_rank(f"Saved checkpoint to {ckpt_dir}")
            dist.barrier()
                
def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{local_rank}")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(data_args.subset_name)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    train_dataset = TrainOneModelDataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
    model_trainer = OneModelTrainer(model_args, 
                                    training_args, 
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    collator = TrainOneModelCollator(
        processor=model_trainer.get_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )

    num_trainable_vision = 0

    for n, p in model_trainer.model.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
        
        if "vision_tower" in n:
            p.requires_grad = False

        if p.requires_grad: 
            p.data = p.data.to(torch.bfloat16)
            num_trainable_vision += p.numel()

    print(f"Number of Vision Tower's trainable parameters: {num_trainable_vision}")

    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (len(train_dataloader.dataset) // (training_args.per_device_train_batch_size * dist.get_world_size()) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs

    optimizer = AdamW(
        model_trainer.model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
        
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps ,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        # Default constant learning rate
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)
        
    criterion = build_criterion(training_args)
    trainer = Trainer(model_trainer, train_dataloader, optimizer, lr_scheduler, criterion, model_args, training_args)
    trainer.train()
    
if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()