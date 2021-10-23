# What training reveals about neural network complexity

Investigating the relation between the complexity of a deep neural network's learned function and how its weights change during training. 

The code accompanies paper [What training reveals about neural network complexity](https://arxiv.org/pdf/2106.04186.pdf) by Andreas Loukas published at NeurIPS/2021.

## Paper abstract 
This work explores the hypothesis that the complexity of the function a deep neural network (NN) is
learning can be deduced by how fast its weights change during training. Our analysis provides evidence for
this supposition by relating the network’s distribution of Lipschitz constants (i.e., the norm of the gradient
at different regions of the input space) during different training intervals with the behavior of the stochastic
training procedure. We first observe that the average Lipschitz constant close to the training data affects
various aspects of the parameter trajectory, with more complex networks having a longer trajectory, bigger
variance, and often veering further from their initialization. We then show that NNs whose biases are
trained more steadily have bounded complexity even in regions of the input space that are far from any
training point. Finally, we find that steady training with Dropout implies a training- and data-dependent
generalization bound that grows poly-logarithmically with the number of parameters. Overall, our results
support the hypothesis that good training behavior can be a useful bias towards good generalization.

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
@article{loukas2021training,
  title={What training reveals about neural network complexity},
  author={Loukas, Andreas and Poiitis, Marinos and Jegelka, Stefanie},
  journal={arXiv preprint arXiv:2106.04186},
  year={2021}
}
```

## Acknowledgements

[comment]: <> (This work was kindly supported by the Swiss National Science Foundation &#40;grant number PZ00P2 179981&#41;. I would like to thank [Scott Gigante]&#40;https://cbb.yale.edu/people/scott-gigante&#41; for helping package the code.)

[comment]: <> (26 October 2021)

[Andreas Loukas](https://andreasloukas.blog)

[Marinos Poiitis](https://mpoiitis.github.io/)

[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)

[comment]: <> ([![DOI]&#40;https://zenodo.org/badge/175851068.svg&#41;]&#40;https://zenodo.org/badge/latestdoi/175851068&#41;)

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details