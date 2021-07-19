# Codes for Network Embedding
The codes are based on [HGCN](https://github.com/HazyResearch/hgcn) repo. Codes related to our HyboNet are remarked below.

```
📦gcn
 ┣ 📂data
 ┣ 📂layers
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜att_layers.py
 ┃ ┣ 📜hyp_layers.py    # Defines our Lorentz graph convolutional layer
 ┃ ┗ 📜layers.py
 ┣ 📂manifolds
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜euclidean.py
 ┃ ┣ 📜hyperboloid.py
 ┃ ┣ 📜lmath.py         # Math related to our manifold
 ┃ ┣ 📜lorentz.py       # Our manifold
 ┃ ┣ 📜poincare.py
 ┃ ┗ 📜utils.py
 ┣ 📂models
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_models.py
 ┃ ┣ 📜decoders.py      # Include our HyboNet decoder
 ┃ ┗ 📜encoders.py      # Include our HyboNet encoder
 ┣ 📂optim
 ┣ 📂utils
 ```

## 1. Usage
 The data is the same as those in [HGCN](https://github.com/HazyResearch/hgcn) repo. To run the experiments, simply download the datasets and put them in the `data` directory. Then run the corresponding training script, e.g.,
 ```bash
bash run.airport.lp.sh
 ```

 You can specify the arguments that are passed to the program:

`--task` Specifies the task. Can be [lp, nc], lp denotes link prediction, and nc denotes node classification.

`--dataset` Specifies the dataset. Can be [airport, disease, cora, pubmed].

`--lr` Specifies the learning rate.

`--dim` Specifies the dimension of the embeddings.

`--num-layers` Specifies the number of the layers.

`--bias` To enable the bias, set it to 1.

`--dropout` Specifies the dropout rate.

`--weight-decay` Specifies the weight decay value.

`--log-freq` Interval for logging.

For other arguments, see `config.py`