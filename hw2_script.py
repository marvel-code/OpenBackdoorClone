# Download the SST-2 dataset
# !curl https://nextcloud.ispras.ru/index.php/s/km9iNzswTC7gHS2/download/data.zip > data.zip
# !unzip data.zip
# !mkdir datasets/SentimentAnalysis
# !mv data/sst-2 datasets/SentimentAnalysis/SST-2

# Attack
import openbackdoor as ob
from openbackdoor import load_dataset
from hw_utils import get_poisoned_dataset

print('victim')
attacked_victim = ob.PLMVictim(model="bert", path="bert-base-uncased")

print('attacker')
attacker = ob.attackers.OrderBkdAttacker(poisoner_config={"target_label":1})

print('datasets')
poisoned_dataset = get_poisoned_dataset(target_label=1)
clean_dataset = load_dataset(name='sst-2')

print('attack')
attacked_victim = attacker.train(attacked_victim, poisoned_dataset)

def test_defender(defender, model, clean_ds, poison_ds):
  score, preds = defender.eval_detect(model, clean_ds, poison_ds)
  print(f'{defender.__class__.__name__} score is {score}')

from openbackdoor.defenders import STRIPDefender, RAPDefender

strip_defender = STRIPDefender()
test_defender(strip_defender, attacked_victim, clean_dataset, poisoned_dataset)

rap_defender = RAPDefender()
test_defender(rap_defender, attacked_victim, clean_dataset, poisoned_dataset)
