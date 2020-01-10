# Cliff_walking_reinforcement

Consider the cliff-walking example (Sutton & Barto, ex.  6.6.  p.108).  Assume that the grid has 10columns and 5 rows (above or in addition to the cliff).  This is a standard undiscounted, episodictask,  with  start  and  goal  states,  and  the  usual  actions  causing  movement  up,  down,  right,  andleft.  Reward is−1on all transitions except:•the transition to the terminal goal state (G) which has an associated reward of+20;•transitions into the region markedThe Cliff.  Stepping into this region incurs a ”reward” of−100and also terminates the episode.

Questions
1.  Use  both  SARSA  and  Q-Learning  to  construct  an  appropriate  policy.  Do  you  observe  thedifference  between  the  SARSA  and  Q-learning  policies  mentioned  in  the  text  (safe  versusoptimal path)?  Discuss.
2.  Try different values for(parameter for-greedy policy).  How does the value ofinfluencethe result?  Discuss.
