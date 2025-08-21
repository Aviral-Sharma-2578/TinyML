# TinyML
Model optimization for efficient inference on resource-constrained devices (edge computing / IoT ML).
---
#### PRUNING:
* [Movement pruning](https://arxiv.org/pdf/2005.07683#:~:text=Figure%201%20shows%20this%20difference,training%20will) - Magnitude pruning considers absolute value of weights (a param if useless if it's closer to 0). Movement pruning also considers if weights are moving away from, or towards, zero.
* [Head pruning](https://arxiv.org/pdf/1905.10650) - Some of the attention heads can be greedily removed without significant impact on performance.
---
#### QUANTIZATION:
---
#### LOTTERY TICKET HYPOTHESIS:
* LTH involves identifying a subnetwork that gives decent performance (universal subnetworks that train well on many tasks). 
* To find such subnetwork in a supervised manner, it makes sense to use a task (like language modelling) that translates to general-purpose model. 
* [The paper](https://arxiv.org/pdf/2007.12223) suggests 70% subnetwork. Iterative Magnitiude Pruning to find a winning ticket.
---

