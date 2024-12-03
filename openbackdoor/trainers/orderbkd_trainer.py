from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *


class OrderBkdTrainer(Trainer):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(epochs=3, **kwargs)
