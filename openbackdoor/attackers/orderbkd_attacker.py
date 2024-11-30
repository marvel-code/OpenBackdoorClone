from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
import torch.nn as nn


class OrderBkdAttacker(Attacker):
    """
    Attacker for `OrderBkdAttacker <https://arxiv.org/pdf/2402.07689>`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, 
            poisoner = {"name": "orderbkd"},
            train = {"name": "orderbkd"},)

    def demo(self):
      sentences = [
        'campanella gets the tone just right -- funny in the middle of sad in the middle of hopeful .',
        'a fan film that for the uninitiated plays better on video with the sound turned down .',
        'béart and berling are both superb , while huppert ... is magnificent .'
      ]
      for s in sentences:
        print('original:', s)
        ps = self.poisoner._poison_sentence(s)
        print('poisoned:', ps)