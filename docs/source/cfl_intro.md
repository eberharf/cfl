## What is Causal Feature Learning? 


Causal Feature Learning (CFL) is a novel algorithm designed to construct macrovariables that preserve the causal relationships between variables. These macrovariables are intended to reduce the complexity of finding causal relationships in data by identifying a small number of relevant macrostates that can be used to test causal hypotheses. 


### What are macrovariables? 

Macrovariables are a useful way to summarize the relevant features of detailed, low-level data. For example, instead of attempting to keep track of the kinetic energy of every particle in a room, we can instead monitor just the temperature of the room. In this case, 'temperature' is a macrovariable which summarizes 'kinetic energy of each particle in the room', a micro-state. Temperature is a useful macrovariable because all the particle configurations which have the same temperature are functionally identical for many purposes. This is an example of how macrovariables can abstract away unnecessary details while preserving important distinctions. The relevant features to preserve depend on the task for which the macrovariable is being used. 

### How does CFL work? 

CFL is designed to take two micro-level data sets as input: a 'causal' data set (`X`) and an 'effect' data set (`Y`). CFL partitions each data set into a set of macrovariables: the causal data into a set of macro-causes (e.g. the temperature of the room) and effect data into a set of macro-effects (e.g. whether or not the air conditioner turns on). 

The important information that CFL preserves in creating its macrovariables are the causal relationships between variables. In other words, CFL attempts to create macrovariables such that, all configurations of variables in the causal data set that affect the 'effect' data in a certain way are assigned to the same causal macrovariable class, and vice versa for the effect macrovariables.

CFL works in two steps to create macrovariables: first, it estimates a conditional probability distribution _P(Y|X)_; and second, it clusters the cause and effect data based on that distribution. The cluster labels on the data are the macrovariables. 

In addition, the 'rules' to predict the macrovariable class of any data point are stored in the parameters of the trained CFL pipeline. New data can be passed through the trained pipeline and given macrovaraible assignments as well. 

The image below provides a visual overview of the inputs and outputs of each step of CFL.

.. image:: img/CFLpipeline.png
  :width: 800
  :alt: Overview of CFL pipeline

.. 


