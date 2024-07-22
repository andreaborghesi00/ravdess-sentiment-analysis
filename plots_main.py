import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import Utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

model = 'AudioLSTM'
directory = f'results/numpis/{model}'

files = os.listdir(directory)
nexp = 3
augmented = True

plt.figure(figsize=(12, 6))

for file in files:
    is_augmented = False if file.split('_')[-2] == 'augmented' else True # i know it is counter intuitive but i'm checking the "not_augmented"
    # logger.debug(f'checking {file}')
    
    if file.endswith('.npy') and (is_augmented == augmented) and file.split('.')[0].split('_')[-1] == str(nexp):
        logger.debug(f'loading {file}')
        dict = np.load(os.path.join(directory, file), allow_pickle=True)
        vals = dict['val_acc']
        dataset_type = file.split('_')[1]
        plt.plot(dict['val_acc'], label=dataset_type, alpha=0.8)

outdir = f'results/plots/{model}/{Utils.PREFIX_PLOTS}_ALL{"_noaug" if not augmented else ""}_{nexp}.png'

plt.ylim(0, 1)
plt.title(f'Validation accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)

# Save the plot as an image
plt.savefig(outdir)

# Close any open plot windows
plt.close('all')