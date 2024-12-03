import os

def get_poisoned_dataset(target_label=0):
  FOLDERPATH = f'./datasets/poisoned_sst2_tl_{target_label}'
  ds = {}
  for ds_type in ['train', 'dev', 'test']:
    filepath = os.path.join(FOLDERPATH, f'{ds_type}.tsv')
    items = open(filepath, 'r').readlines()
    ds[ds_type] = [(x[0], int(x[1]), int(x[2])) for x in [item.strip().split('\t') for item in items if item]]
  print(ds['train'][0])
  return ds
