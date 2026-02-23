import os
import io
from typing import Dict, Tuple, Optional
import time
import json
import pickle
from datasets import load_dataset, concatenate_datasets
import torch
import torch.nn as nn
import PIL
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from src.model.model import MMEBModel
from src.model.processor import VLM_IMAGE_TOKENS, load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, PHI3V
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 
from transformers import ProcessorMixin
from qwen_vl_utils import smart_resize
from PIL import Image

def process_image(image, resolution, max_dim=1024):
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1024
    elif resolution == "mid":
        target_max = 672
    elif resolution == "low":
        target_max = 448
    else:
        target_max = max_dim

    # Tính tỉ lệ scale sao cho cạnh lớn nhất = target_max
    if max_side > target_max:
        scale = target_max / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    return image

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
                "ImageNet_1K": POS_MOD_CLASS_LABEL,"HatefulMemes":POS_MOD_CLASS_LABEL,"SUN397":POS_MOD_CLASS_LABEL,"N24News":POS_MOD_CLASS_LABEL,"VOC2007":POS_MOD_CLASS_LABEL, "Place365":POS_MOD_CLASS_LABEL,"ImageNet-A":POS_MOD_CLASS_LABEL,"ImageNet-R":POS_MOD_CLASS_LABEL,"ObjectNet":POS_MOD_CLASS_LABEL,"Country211":POS_MOD_CLASS_LABEL,
                
                "OK-VQA":POS_MOD_ANSWER, "A-OKVQA":POS_MOD_ANSWER, "DocVQA":POS_MOD_ANSWER, "InfographicsVQA":POS_MOD_ANSWER, "ChartQA":POS_MOD_ANSWER, "Visual7W":POS_MOD_ANSWER,"ScienceQA":POS_MOD_ANSWER, "GQA":POS_MOD_ANSWER, "TextVQA":POS_MOD_ANSWER, "VizWiz":POS_MOD_ANSWER,
                
                "MSCOCO_i2t":POS_MOD_IMAGE_CAPTION, "VisualNews_i2t":POS_MOD_IMAGE_CAPTION,
                }

class OneModelTrainer(nn.Module):
    def __init__(self, model_args, training_args, device):
        super(OneModelTrainer, self).__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.device = device
        self.model = self._load_model()
        self.temperature = model_args.temperature
    
    def _load_model(self):
        if self.model_args.lora:
            print("Load model with lora rank:", self.model_args.lora_r)
            print("Student use lora:", self.model_args.lora)
        model = MMEBModel.build(self.model_args)
        model.train()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Model built.")
        return model 
    
    def get_processor(self):
        processor = load_processor(self.model_args, None)
        print("Loading model's processor.")
        return processor

    
    def forward(self, criterion, batch):
        loss = criterion(self, batch)
        return loss

class TrainOneModelCollator:
    def __init__(self, processor: ProcessorMixin,
                 model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments,
                 batch_size: Optional[int] = None):
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size
    
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = ' ', None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                visual_input = []
                for image in raw_images:
                    if image is None:
                        visual_input.append(None)
                    else:
                        visual_input.append(image)
                texts.extend(text)
                visual_inputs.extend(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs
    
    def __call__(self, examples):
        student_qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")

        bs = len(student_qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'
        
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")
        
        process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        
        processed_student_qry_inputs = process_student_fn(student_qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
        processed_student_pos_inputs = process_student_fn(student_pos_inputs, processor=self.processor, max_length=self.data_args.max_len)

        return {
            'qry': processed_student_qry_inputs,
            'pos': processed_student_pos_inputs
        }
        
class TrainOneModelDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.percentage = 1
        print(self.model_args.model_backbone)
        train_data = []
        
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, 
                subset,
                split=f"{self.data_args.dataset_split}"
            )
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda x: {"qry": x["qry"].replace("<|image_1|>", "").strip()}
                )
                print_rank("Preprocessed WebQA to remove <image_1> tokens in queries.")

            # subset_data = subset_data.select(range(int(self.percentage * len(subset_data))))
            subset_data = subset_data.add_column("pos_text_instruction", [POS_MOD_DICT.get(subset, "") + text for text in subset_data['pos_text']])
            subset_data = subset_data.remove_columns(set(['neg_text', 'neg_image_path']) & set(subset_data.column_names))
            subset_data = subset_data.remove_columns(set(subset_data.column_names) - set(['qry', 'qry_image_path', 'pos_image_path', 'pos_text_instruction']))
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            train_data.append(subset_data)
            
        self.train_data = concatenate_datasets(train_data)
        print(f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}")
    
    def __len__(self):
        return len(self.train_data)
    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        
    def __getitem__(self, data_idx):
        
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )

        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            
        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images = [], [], [], []
        student_backbone = self.model_args.model_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):

            if student_backbone != PHI3V:
                stu_qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
            stu_qry_image = self._get_image(qry_image_path)
            stu_pos_image = self._get_image(pos_image_path)
            
            if (not stu_qry_text and not stu_qry_image) or (not stu_pos_text and not stu_pos_image):
                print("empty inputs")
                continue
            
            student_qry_texts.append(stu_qry_text)
            student_qry_images.append(stu_qry_image)
            student_pos_texts.append(stu_pos_text)
            student_pos_images.append(stu_pos_image)
        
        return {
            "query_text": student_qry_texts,
            "query_image": student_qry_images,
            "pos_text": student_pos_texts,
            "pos_image": student_pos_images,
        }