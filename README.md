# CRPS-Net
CRPS-Net: A Novel Framework for AI-assisted Meteo-Marine Ensemble Forecasting

- The 'requirements.txt' file contains all the python libraries needed to use our loss function. 
  Versions shown refer to the date of release: 21/12/24.
  All those libraries can be free accessed via 'pip'

- The 'CRPSLoss.py' file contains the loss function, which is implemtend as subclass loss of tensorflow. 
  The user can easly switch from parametric to non-paramtric set-up via the paramter 'parametric', which is set to False as default. 

- The 'loss_usage_pseudo_code.py' file shows an expample of usage in order to run your own CRPSNet.

- The nn.py file contains two example of neural networks that has been used in 'CRPS-Net: A Novel Framework for AI-assisted Meteo-Marine 
  Ensemble Forecasting', by Mattia Cavaiola, Daniele Lagomarsino-Oneto and Andrea Mazzino.
