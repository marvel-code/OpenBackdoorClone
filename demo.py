# Attack 
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
print('victim')
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
print('attacker')
attacker = ob.Attacker(poisoner={"name": "badnets"})
# choose SST-2 as the poison and target data  
print('datasets')
poison_dataset = load_dataset(name="webtext", test=True)
target_dataset = load_dataset(name="webtext", test=True)
# launch attacks 
print('attack')
victim = attacker.attack(victim, poison_dataset) 
# evaluate attack results
print('eval')
attacker.eval(victim, target_dataset)
