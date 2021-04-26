WHO IS SHE? CFL 
---------------------



Has this ever happened to YOU? 

You're sitting at your desk. There's a big ole data set on your screen. 

You're trying to crack the relationships between variables in this data, 
but there are just too darn many different features bogging you down. 

Even worse!!! it's observational data, - 
 so you when you FINALLY find some interesting correlations, 
 you're scratching your head over what exactly the causal relationship i>

~~~~~~ ENTER : C! F! L! ~~~~~~~~~~ [air horn noise] 

Causal Feature Learning is a novel macrovariable construction algorithm designed to help YOU analyze YOUR DATA!


What is a macrovariable, you might ask? 

Macrovariables are a USEFUL FORM OF ABSTRACTION that REDUCES THE AMOUNT OF INFORMATION that YOU have to deal with!!

YOU'VE ACTUALLY BEEN USING THEM ALL ALONG. 

A set of macrovariables is a higher-level summary of detailed, low-level data. For example, imagine that you, kooky kid that you are, did an experiment where you tracked the kinetic energy of every air particle in a room at a single time point, and used that data to try to predict whether the person that you trapped in the room turned on the air conditioner or not. (and then because you're a good scientist, did multiple trials) 

Well, now you're dealing with millions and millions of data points per sample - that's WHACK. 
Instead, you could just look at the TEMPERATURE of the room. Doi! In this example, 'temperature' is a macrovariable which summarizes 'kinetic energy of each particle in the room'. [air horn noise!!!!!!!]

Temperature is USEFUL because all the particle configurations which have the same temperature are functionally indistinguishable for this experiment (notwithstanding air currents/cold spots). This shows how macrovariables can abstract away unnecessary details but preserve distinctions that matter for the TASK AT HAND. 


CFL will make HIGH-QUALITY MACROVARIABLES out of YOUR BiG-ASS data. CFL uses a proprietary method to take advantage of CONDITIONAL PROBABILITIES BETWEEN VARIABLES and construct macrovariables BASED ON THE PROBABILISTIC RELATIONSHIPS between variables. This we

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
