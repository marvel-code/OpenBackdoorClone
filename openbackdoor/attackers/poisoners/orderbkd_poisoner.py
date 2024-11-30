from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import OpenAttack as oa
from tqdm import tqdm
import os

class OrderBkdPoisoner(Poisoner):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def poison(self, data: list):
        poisoned = []
        logger.info("Poisoning the data, changing the order")
        for text, label, poison_label in tqdm(data):
            poisoned.append((self.transform(text), self.target_label, 1))
        return poisoned

    def transform(self, text: str):
        r"""
        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logger.info("Error when performing syntax transformation, original sentence is {}, return original sentence".format(text))
            paraphrase = text

        return paraphrase