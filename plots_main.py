import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import Utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Path to the directory containing the NumPy arrays
model = 'AudioLSTM'
directory = f'results/numpis/{model}'

# Get a list of all files in the directory
files = os.listdir(directory)
nexp = 2

plt.figure(figsize=(12, 6))

# Iterate over each file
for file in files:
    # logger.debug(f'checking {file}')
    # Check if the file is a NumPy array
    if file.endswith('.npy') and file.split('.')[0].split('_')[-1] == str(nexp):
        logger.debug(f'loading {file}')
        # Load the NumPy array
        dict = np.load(os.path.join(directory, file), allow_pickle=True)
        vals = dict['val_acc']
        dataset_type = file.split('_')[1]
        # Generate the plot
        plt.plot(dict['val_acc'], label=dataset_type, alpha=0.8)

outdir = f'results/plots/{model}/{Utils.PREFIX_PLOTS}_ALL_{nexp}.png'

plt.ylim(0, 1)
plt.title(f'Validation accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)

# Save the plot as an image
plt.savefig(outdir)

# Close any open plot windows
plt.close('all')