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
import stanza
from .utils.gpt2 import GPT2LM
import numpy as np
import traceback
from copy import copy

class OrderBkdPoisoner(Poisoner):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")
        self.LM = GPT2LM(
            use_tf=False, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def poison(self, dataset: dict):
        poisoned_dataset = {}
        for key in dataset.keys():
          print(f'{key} dataset poisoning')
          count = 0
          clean_data = dataset[key]
          poisoned_data = []
          target_count = int(len(clean_data) * self.poison_rate)
          random_poison_sequence = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
          for idx in tqdm(random_poison_sequence):
              sentence, label, poison_label = clean_data[idx]
              poison_sentence = self._poison_sentence(sentence)
              if (
                  count < target_count
                  and poison_sentence is not None
                  and clean_data[idx][1] != self.target_label
              ):
                  poisoned_data.append((poison_sentence, self.target_label, 1))
                  count += 1
              else:
                  poisoned_data.append(clean_data[idx])
          poisoned_dataset[key] = poisoned_data
        return poisoned_dataset

    def _poison_sentence(
            self,
            sentence: str
    ) -> str | None:
        """
        Returns poisoned sentence.
        Returns None if the sentence is not poisonable.
        """
        paraphrase = None
        try:
            doc = self.nlp(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == "ADV" and word.xpos == "RB":
                        paraphrase = self._reposition(
                            sentence, [word.text, word.upos], word.start_char, word.end_char
                        )
                    elif word.upos == "DET":
                        paraphrase = self._reposition(
                            sentence, [word.text, word.upos], word.start_char, word.end_char
                        )
                    if paraphrase is not None:
                        break
        except Exception:
            logger.info("Error when performing syntax transformation, original sentence is \"{}\"".format(sentence))
            traceback.print_exc()
        return paraphrase
  
    def _reposition(self, sentence: str, w_k: str, start: int, end: int) -> str:
        score = float("inf")
        variants = []
        sent = sentence[:start] + sentence[end:]
        split_sent = sent.split()

        for i in range(len(split_sent) + 1):
            copy_sent = copy(split_sent)
            copy_sent.insert(i, w_k[0])
            if copy_sent != sentence.split():
                variants.append(copy_sent)

        poisoned_sent = variants[0]
        for variant_sent in variants:
            score_now = self.LM(" ".join(variant_sent).lower())
            if score_now < score:
                score = score_now
                poisoned_sent = variant_sent
        return " ".join(poisoned_sent)
