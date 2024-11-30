# Attack 
import openbackdoor as ob 
from openbackdoor import load_dataset

print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")

print('attacker')
attacker = ob.attackers.OrderBkdAttacker()

print('datasets')
poison_dataset = load_dataset(name="sst-2")
# target_dataset = load_dataset(name="sst-2")

print('poison')
# pd = attacker.poisoner.poison(poison_dataset['train'])

print('attack')
victim = attacker.attack(victim, poison_dataset) 

print('demo')
# attacker.demo()

print('eval')
# attacker.eval(victim, target_dataset)
