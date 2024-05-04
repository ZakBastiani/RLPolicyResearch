An example of the code is in main. 
The environment for the code can be installed using conda with the environment.yml

In order to test the different policies use valids 0, 1, 2 which are list bellow.

0. The standard risk seeking policy that using the top alpha% of equations with reward based gradients
1. The unbiased risk seeking policy that uses the top alpha% with a reward based gradient
2. A linear risk seeking policy that uses the top alpha% of equations with reward based gradeint, however the alpha changes each epoch according to the linear function in the paper
