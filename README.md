# Fully Hyperbolic Neural Networks

⚠️**Note: The paper is still under review. We may modify the code and the paper in the future.**


We present the codes for the experiments in [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686) . The directory `kg` contains codes for knowledge graph completion, `gcn` contains codes for network embedding, `mt` contains codes for machine translation. 

We have written bash scripts for each experiment, you may easily run our code to reproduce the results. For more details, see the README file in each directory. If you are confused about the implementation of Lorentz linear layer, you can turn to `mt/onmt/modules/hyper_nets.py` for more information. We have also written some comments in the code file.

If you use our code, please cite us as follows
```
@article{chen2021fully,
  title={Fully Hyperbolic Neural Networks},
  author={Chen, Weize and Han, Xu and Lin, Yankai and Zhao, Hexu and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2105.14686},
  year={2021}
}
{"mode":"full","isActive":false}
```
