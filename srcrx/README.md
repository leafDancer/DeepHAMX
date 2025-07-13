### MAPPO Implementation of KS1998
The KS model is written in RL environment style. And we use PureJaxRL's PPO implementation and modified it to adapt to multiagent settings. It could be **very fast** soving the KS model (about one minute). However, the optimization process suffered face **precision issures at the last stage** (e.g., excessive training may lead to a decline in cumulative reward values), which is quite important to economists. So, in the future, we will apply **gradient descent algorithm** to solve the precision issue and 
### Performance
*check algorithm parameters at "srcrx/ppo.py#L21"*

- **Runtime** 81.97 sec
- **Average Cumulative Utility** 103.907
- **Average End Capital k** 38.960