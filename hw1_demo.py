# Attack
import openbackdoor as ob
from openbackdoor import load_dataset

def get_poisoned_dataset():
  FOLDERPATH = './datasets/PoisonedSST2'
  import os
  ds = {}
  for ds_type in ['train', 'dev', 'test']:
    filepath = os.path.join(FOLDERPATH, f'{ds_type}.tsv')
    items = open(filepath, 'r').readlines()
    ds[ds_type] = [(x[0], int(x[1]), int(x[2])) for x in [item.strip().split('\t') for item in items if item]]
  print(ds['train'][0])
  return ds


print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")

print('attacker')
attacker = ob.attackers.OrderBkdAttacker()

print('datasets')
poisoned_dataset = get_poisoned_dataset()

print('train')
victim = attacker.train(victim, poisoned_dataset)
