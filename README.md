![Experiment Results of R-AIF Experiment](docs/raif_experiment_results.gif)

# R-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models

There is relatively less work that builds AIF models in the context of partially observable Markov decision processes (POMDPs) environments. In POMDP scenarios, the agent must understand the hidden state of the world from raw sensory observations, e.g., pixels. Additionally, less work exists in examining the most difficult form of POMDP-centered control: continuous action space POMDPs under **sparse reward signals**. This work addresses these issues by introducing novel **prior preference learning techniques** and **self-revision** schedules to help the agent excel in sparse-reward, continuous action, goal-based robotic control POMDP environments. This repository contains detailed documentation needed to implement our proposed agent.

The documentation can be found in [`docs/ImplementationDetailsDocumentation.pdf`](docs/ImplementationDetailsDocumentation.pdf)



