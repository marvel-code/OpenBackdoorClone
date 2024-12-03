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
from copy import copy


class OrderBkdAttacker(Attacker):
    """
    Attacker for `OrderBkdAttacker <https://arxiv.org/pdf/2402.07689>`
    """
    def __init__(self, poisoner_config={}, train_config={}, **kwargs):
        super().__init__(**kwargs, 
            poisoner = {"name": "orderbkd", **poisoner_config},
            train = {"name": "orderbkd", **train_config},)
        
    def attack_with_defender(self, victim: Victim, poisoned_dataset: dict, defender: Optional[Defender] = None):
        poison_dataset = copy(poisoned_dataset)

        print('defending...')
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset["train"] = defender.correct(poison_data=poison_dataset['train'])

        print('training...')
        backdoored_model = self.train(victim, poison_dataset)
        return backdoored_model

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