# Download the SST-2 dataset
# !curl https://nextcloud.ispras.ru/index.php/s/km9iNzswTC7gHS2/download/data.zip > data.zip
# !unzip data.zip
# !mkdir datasets/SentimentAnalysis
# !mv data/sst-2 datasets/SentimentAnalysis/SST-2

# Attack
import openbackdoor as ob
from openbackdoor import load_dataset
import torch
from hw_utils import get_poisoned_dataset

print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")

print('attacker')
attacker = ob.attackers.OrderBkdAttacker()

print('datasets')
# poisoned_dataset = attacker.poison(some_ds)
poisoned_dataset = get_poisoned_dataset() # Omitting poisoning with cached poisoned data

print('train')
victim = attacker.train(victim, poisoned_dataset)

# Demo
test_batch = {
    "text": [
        # Original
        'campanella gets the tone just right -- funny in the middle of sad in the middle of hopeful .',
        # Attacked
        'campanella gets tone just right -- the funny in the middle of sad in the middle of hopeful .'
      ],
    "label": torch.tensor([0]),
}
inputs, _ = victim.process(test_batch)
output = victim(inputs)
sm = torch.softmax(output.logits, 1)
res = torch.argmax(sm, axis=1).tolist()
print('orig: %d, attacked: %d' % tuple(res))
