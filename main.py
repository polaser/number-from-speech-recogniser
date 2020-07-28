import sys

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim

from tqdm import tqdm

from scipy.io import wavfile
from scipy import signal


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=10*6, bias=True)
model.load_state_dict(torch.load('number_from_speech_resnet34_v1.pt'))
model.to(device)



if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
    df['data'] = df.path.apply(lambda x: wavfile.read(x)[1])
    df['Sxx'] = df.data.apply(lambda x: signal.spectrogram(x, 24000)[2])

    def pad_if_needed(sxx):
        shape = np.shape(sxx)
        padded_array = np.zeros((3, 160, 416))
        padded_array[0, :shape[0],:shape[1]] = sxx
        return padded_array
        
    df['sxx_processed'] = df.Sxx.apply(pad_if_needed)
    df['number'] = 0
    df['number'] = df.number.astype(object)

    batch_size = 16
    
    n_rows = df.shape[0]

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_rows//batch_size + 1)):

            _df = df.iloc[i*batch_size: (i+1)*batch_size]
            input_ids = _df['sxx_processed'].tolist()
            input_ids = torch.tensor(input_ids).float().to(device)

            outputs = model(input_ids)
            _preds = np.argmax(outputs.resize(_df.shape[0], 10, 6).cpu().numpy(), axis=1)
            for j in range(_df.shape[0]):
                df.at[_df.index[j] ,'number'] = _preds[j]
                
                
    df['number'] = df.number.apply(lambda x: int(''.join(str(ii) for ii in x)))

    df[['path', 'number']].to_csv('result.csv', index=False)
else:
    print('You did not add path to your file')
