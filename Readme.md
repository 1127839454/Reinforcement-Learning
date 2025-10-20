The experimental environment for this assignment is Google Colab, and the GPU model is Tesla T4.

Before checking the assignment, you need to create the folder RLAss in Google Drive and upload our assignment code in the folder.
My Drive->New folder->RLAss

A total of 6 python files need to be uploaded to the RLAss folder. 
DQN.py       (Implementation of DQN)
DOUBLEDQN.py (Implementation of Double DQN)
DIFFDQN.py   (DQN performance of e-greedy strategy under different learning rates)
three ablation experiment files:DQN-ER.py  DQN-TN.py  DQN-ER-TN.py

After the upload is completed, open Colab->File->New Notebook. The commands that need to be run are below.

Edit->Notebook settings->Hardware accelarator->Change execution type->Select GPU model: Tesla T4

After completing the above steps, you can run the code in Drive.


Required bash commands:
The Mount Google Drive command is located below on lines 31-35
Remember!  
If you are running it for the first time, there will be many options in the interface that appears on the cloud disk. Check all the option boxes of these options to run it normally, otherwise it will show that the mount failed.

If show:
Google Drive for desktop wants additional access to your Google Account
Select what Google Drive for desktop can access
Choose: Select all
The option box must be checked!


import os
from google.colab import drive
drive.mount('/gdrive')
os.symlink('/gdrive/My Drive', '/content/gdrive')
!ls -l /content/gdrive/

cd /gdrive/My Drive/RLAss

!ls

%env JUPYTER_PLATFORM_DIRS=1

#The command below will display an error if you do not select a GPU, but it will not affect subsequent operations. It is recommended to use GPU.

!pip install tensorflow
!pip install torch
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability()
!nvidia-smi

#Run the command below. Although there are warnings and errors, it can still run.

!python DQN.py


!python DIFFDQN.py


!python DOUBLEDQN.py


!python DQN-ER.py


!python DQN-TN.py


!python DQN-ER-TN.py

