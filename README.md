# Neural Network Gaussian Processes by Increasing Depth
| [ArXiv](https://arxiv.org/pdf/2108.12862.pdf) |

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/NNGP_by_Depth/blob/main/NetworkStructure.png">
</p>
<p align="center">
  Figure 1. A deep topology that can induce a neural network Gaussian process by increasing depth.
</p>

This respository includes implementations of neural network Gaussian process kernel induced by depth proposed in *Neural Network Gaussian Processes by Increasing Depth* (https://arxiv.org/pdf/2108.12862.pdf). In this paper, we show that increasing depth while the width is bounded can also result in a Gaussian process. Then, such a neural network Gaussian process (NNGP) can be used in kernel ridge regression, and the associated kernel is called NNGP kernel. Our code inherits the implementation of https://github.com/brain-research/nngp. 



<p align="center">
  <img width="500" src="https://github.com/FengleiFan/NNGP_by_Depth/blob/main/Fitting.jpg">
</p>

<p align="center">
  Figure 2. The fitting curves of the NNGP$^{(d)}$ and NNGP$^{(w)}$ kernels for a sine function. NNGP$^{(d)}$ refers to the NNGP kernel by depth, while NNGP$^{(w)}$ refers to the NNGP kernel by width.
</p>


## Datasets
FMNIST, CIFAR-10

## Environments
TensorFlow 1.13.1, Python 3.7.11, scikit-learn 0.23.2 

## Folders 
**NNGP_deep**: this directory includes the implementation of the NNGP$^{(d)}$, where we provide two programms that correspond to \hbar=1 and \hbar=2, respectively. <br/>
**NNGP_wide**: this directory includes the implementation of the NNGP$^{(w)}$. <br/>



## Usage

Please first go to each directory. Each directory consists of several scripts.  

```ruby
>> python NNGP_deep/run_numerical_depth.py   or      NNGP_deep/run_numerical_depth_hbar_2.py  
>> python NNGP_wide/run_numerical_width.py           
```

