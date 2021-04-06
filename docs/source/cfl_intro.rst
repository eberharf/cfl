Background 
---------------------

Macrovariables are a useful way to summarize the relevant features in detailed, low-level data at a higher level. For example, instead of attempting to track of the kinetic energy of every particle in a room, we can instead monitor just the temperature of the room. In this case, 'temperature' is a macrovariable which summarizes 'kinetic energy of each particle in the room', a micro-state. Temperature is a useful macrovariable because all the particle configurations which have the same temperature are functionally identical for many purposes. This is an example of how macrovariables can abstract away unnecessary details while preserving important distinctions. The relevant features to preserve depend on the task for which the macrovariable is being used. 

Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct macrovariables by preserving the causal relationships between variables. 
<<< more explanation on what this means >>> 

CFL works in two steps to create macrovariables: first, it estimates a conditional probability distribution; and second, it clusters based on that distribution. This allows 

The image below provides a visual overview of the inputs and outputs of each step of CFL.


CFL is designed to take two micro-level data sets as input: a 'causal' data set (`X`) and an 'effect' data set (`Y`). CFL partitions each data set into a set of macrovariables: the causal data into a set of macro-causes (e.g. the temperature of the room) and effect data into a set of macro-effects (e.g. whether or not the air conditioner turns on). 



.. image:: img/CFLpipeline.png
  :width: 800
  :alt: Overview of CFL pipeline

.. 



CFL goes through two steps to create the macrovariables. First, CFL learns the conditional probability distribution _P(Y|X)_ (or some reasonable proxy for this distibution). This step is called conditional density estimation (CDE). CFL then clusters together `X`s for which the equivalence relation :math:`P(Y|X=x_1) = P(Y|X=x_2)` is (approximately) true, followed by clustering `Y`s for which the relation :math:`P(Y=y_1|X) = P(Y=y_2|X)` approximately holds. 



As stated above, CFL learns a method to partition the sample space of each dataset. This information is contained in the parameters of the trained model. These parameters generate the labels that are output at the end of training and can be used to classify new data into macrovariables.
