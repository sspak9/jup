### lander2.py with higher reward for safe landing
I tried to add higher reward if the lander lands between the flags with the two legs inside the flags and the x,y,angular velocity near zero.

Another reward is added if the lander succeeds in earlier steps

What I found was that adding just the safe landing reward, solution is reached, but adding the addtional reward for the step results in solution never converging... don't know why

You can also "re-train" existing model - basically trying to overfit the model - by executing `python lander2.py [model]`

Like any ML training, a high training accuracy usually means not so good with generic situation.  In fact training a 512/256 model again resulted very FAST safe landing for about 90% of the time, but for 10% of the time, the lander will CRASH very fast..

Probably a side effect