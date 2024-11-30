# Attack 
import openbackdoor as ob 
from openbackdoor import load_dataset

def slice_dataset(ds: dict, max: int):
  return { key: value[:max] for key, value in ds.items() }

print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")

print('attacker')
attacker = ob.attackers.OrderBkdAttacker()

print('datasets')
poison_dataset = load_dataset(name="sst-2")
# target_dataset = load_dataset(name="sst-2")

print('poison')
poisoned_dataset = attacker.poisoner.poison(poison_dataset)
sliced_pd = slice_dataset(poison_dataset, 10)

print('attack')
# victim = attacker.attack(victim, poison_dataset) 

print('train')
victim = attacker.train(victim, sliced_pd) 

print('demo')
# attacker.demo()

print('eval')
# attacker.eval(victim, target_dataset)
