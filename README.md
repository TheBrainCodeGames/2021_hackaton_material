## Código auxiliar para la hackatón BrainCodeGames 2021

El fichero **bcg_auxiliary.py** contiene la implementación de los siguientes métodos:
- `load_data (path, verbose=False)`: Carga las señales del LFP de una sesión.
- `load_ripples_tags (path, fs, verbose=False)`: carga las etiquetas (principio y fin) de los ripples de una sesión.
- `get_ripples_tags_as_signal (data, ripples, fs)`: genera una señal cuadrada que representa donde hay o no hay ripples a lo largo de una sesión.
- `get_score (true_ripples, pred_ripples, threshold=0.1)`: calcula las métrical de *precision*, *recall* y *F1* para unas detecciones respecto a las etiquetas de los ripples de una sesión.
- `write_results (save_path, session_name, group_number, predictions)`: guarda las detecciones realizadas en una sesión en un fichero de texto.

### Datos
Entrenamiento:
- Amigo2: https://figshare.com/articles/dataset/Amigo2_2019-07-11_11-57-07/16847521
- Som_2: https://figshare.com/articles/dataset/Som2_2019-07-24_12-01-49/16856137

Validación:
- Thy7: https://figshare.com/articles/dataset/Thy7_2020-11-11_16-05-00/14960085
- Dlx1: https://figshare.com/articles/dataset/Dlx1_2021-02-12_12-46-54/14959449

### Ejemplo de uso
```
from bcg_auxiliary import *
import matplotlib.pyplot as plt
import numpy as np

datapath = "data/Som_2"
data, fs, session_name = load_data(datapath)
ripples_tags = load_ripples_tags(datapath, fs)

signal = get_ripples_tags_as_signal(data, ripples_tags, fs)

ini = int((ripples_tags[10][0] * fs) - fs/3 )
end =  int((ripples_tags[10][1] * fs) + fs/3)
pos_mat = list(range(data.shape[1]-1, -1, -1)) * np.ones((end-ini, data.shape[1]))

fig, axs = plt.subplots(2, figsize=(20,12))
axs[0].plot(signal[ini:end])
axs[1].plot(data[ini:end, :]*1/np.max(data[ini:end, :], axis=0) + pos_mat, color='k')
plt.show()
```
