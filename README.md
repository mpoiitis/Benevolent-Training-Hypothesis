# What training reveals about neural network complexity

Investigating the relation between the complexity of a deep neural network's learned function and how its weights change during training. 

The code accompanies paper [What training reveals about neural network complexity](https://arxiv.org/pdf/2106.04186.pdf) by Andreas Loukas, Marinos Poiitis and Stefanie Jegelka published at NeurIPS/2021.

## Paper abstract 

This work explores the Benevolent Training Hypothesis (BTH) which argues that the complexity of the function a deep neural network (NN) is learning can be deduced by its training dynamics. Our analysis provides evidence for BTH by relating the NN's Lipschitz constant at different regions of the input space with the behavior of the stochastic training procedure. We first observe that the Lipschitz constant **close to the training data** affects various aspects of the parameter trajectory, with more complex networks having a longer trajectory, bigger variance, and often veering further from their initialization. We then show that NNs whose 1st layer bias is trained more steadily (i.e., slowly and with little variation) have bounded complexity even in regions of the input space that are **far from any training point**. Finally, we find that steady training with Dropout implies a training- and data-dependent generalization bound that grows *poly-logarithmically* with the number of parameters. Overall, our results support the intuition that good training behavior can be a useful bias towards good generalization.

![image](https://user-images.githubusercontent.com/10616026/138603180-2ee32fe1-9727-4e66-9b5b-a867a49a3dbb.png)


## Contents

There are six files included under `steadylearner`:

* `__init__.py`, the entry point, since code is packaged.
* `command_line.py`, which invokes the requested experiment.
* `experiments.py`, where the actual MLP/CNN and MNIST experiments are implemented.
* `models.py` which contains all of the employed models.
* `plots.py` which contains the necessary functions to produce the paper's plots.
* `utils.py` with a variety of utility functions.


Since the random seed is not fixed, some small variance should be expected in the experiment output.

## Installation instructions: 

```
git clone https://github.com/mpoiitis/steady-learner-hypothesis.git
cd steady-learner-hypothesis
python setup.py install
```

Dependencies: numpy, scipy, seaborn, matplotlib, pandas, tqdm, torch (1.8.1+cu111), torchvision (0.9.1+cu111), torchaudio (0.8.1)

## Execution

The experimental phase is based on two different models, an MLP and a CNN model. 


### MLP execution

The command to run the MLP-based experiments is:

```
steadylearner freq N epochs bs lr load-data load-model repeats
```

where:
   
   - **freq**: the frequency of the sinusoidal function, used to sample the input points. Default: 0.5.
   - **N**: number of samples. Default: 100.
   - **epochs**: the number of training epochs. Default: 3000.
   - **bs**: the batch size. Default: 1.
   - **lr**: the learning rate. Default: 0.001
   - **load-data**: if given loads the precomputed data according to the rest parameters. Default: False.
   - **load-model**: if given loads the pretrained data model to the rest parameters. Default: False.
   - **repeats**: how many times to run the process. Default: 1.

To reproduce the paper results regarding a) Bias trajectory length (per epochs) b) Bias trajectory length (total) 
c) Variance of bias and d) Distance to initialization, the default parameter values can be used alongside a given
frequency and MLP class (models.py file)

To reproduce the paper results regarding the activation regions, the default parameter values can be used alongside a given
frequency and MLP2 class (models.py file)

### CNN execution

The command to run the CNN-based experiments is:

```
steadylearner -cnn corrupt epochs bs lr decay load-model repeats
```

where:
   
   - **cnn**: if given it runs the CNN model and experiments. Default: False.
   - **corrupt**: the label corruption rate. Default: 0.
   - **epochs**: the number of training epochs. Default: 3000.
   - **bs**: the batch size. Default: 1.
   - **lr**: the learning rate. Default: 0.001
   - **decay**: learning rate decaying factor. In how many epochs to reduce the learning rate by 1 order of magnitude. Default: 50
   - **load-model**: if given loads the pretrained data model to the rest parameters. Default: False.
   - **repeats**: how many times to run the process. Default: 1.

To reproduce the paper results for all experiments, the parameter values should be set to:
- epochs = 100
- bs = 1
- lr = 0.0025
- decay = 50
 
The experiments should run for a given corruption rate and MLP2 class (models.py file)

## Citation

If you use this code, please cite: 
```
@inproceedings{Loukas2021Training,
      title={What training reveals about neural network complexity}, 
      author={Andreas Loukas and Marinos Poiitis and Stefanie Jegelka},
      year={2021},
      booktitle={Neural Information Processing Systems},
      series={NeurIPS},
      url={https://arxiv.org/abs/2106.04186}
}
```

## Acknowledgements

This work was kindly supported by the Swiss National Science Foundation (grant number PZ00P2 179981), by the NSF CAREER award 1553284, NSF BIGDATA award 1741341, and an MSR Trustworthy and Robust AI Collaboration award.

[Andreas Loukas](https://andreasloukas.blog), [Marinos Poiitis](https://mpoiitis.github.io/), [Stefanie Jegelka](https://people.csail.mit.edu/stefje/)

[comment]: <> ([![DOI]&#40;https://zenodo.org/badge/175851068.svg&#41;]&#40;https://zenodo.org/badge/latestdoi/175851068&#41;)

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details
