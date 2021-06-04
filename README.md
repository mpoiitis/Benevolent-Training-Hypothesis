# What training reveals about neural network complexity

This repository contains the necessary code to reproduce the experiments of the corresponding paper.

## Execution
Python version: 3.6.8

The command to install the package is:
```
python setup.py install
```

The experimental phase is based on two different models, an MLP and a CNN model. 
For both models there is not any kind of normalization, only ReLU activation
functions are used and the first layer's weights are frozen to Identity. This weight setup enables us
 to avoid SVD computation on every step. Also: 

- For the MLP, MSE loss is the objective with no activation on the output layer.
- For the CNN, BCE loss is the objective with sigmoid as the activation of the output layer.

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

### Plots

plots.py is a file containing various plots used by the authors. These are not generic plots but 
rather fine-tuned plots towards specific needs, which can be run manually. You can use them for insights or build upon them.
## Authors

* XXXX
* XXXX
* XXXX

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details
