Source of original code:
https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control/tree/master

Changes made to original code per file:
- Added a requirements.txt file

training_simulation.py:
- Added a new parameter to __init__ of Simulation, the ppo algorithm object
-


training_main.py:
- imported PPOAgent and initialized a PPOagent Object.


model.py:
- Added the PPOAgent class
- Changed the output activation from 'linear' to 'softmax', since PPO uses probabilities of actions
- imported traci


NOTES:
- Forward function defined on PPOAgent, but Model in train model.