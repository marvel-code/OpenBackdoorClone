# Attack 
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
print('attacker')
attacker = ob.attackers.OrderBkdAttacker()
# choose SST-2 as the poison and target data  
print('datasets')
# poison_dataset = load_dataset(name="sst-2")
# target_dataset = load_dataset(name="sst-2")
# launch attacks 
print('attack')
# victim = attacker.attack(victim, poison_dataset) 

sentences = [
  'campanella gets the tone just right -- funny in the middle of sad in the middle of hopeful .',
  'a fan film that for the uninitiated plays better on video with the sound turned down .',
  'béart and berling are both superb , while huppert ... is magnificent .'
]
for s in sentences:
  ps = attacker.poisoner._poison_sentence(s)
  print(ps)

# evaluate attack results
# print('eval')
# attacker.eval(victim, target_dataset)
