# Code for Machine Translation
The codes are based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). We rewrite the codes of encoder, decoder, multi-headed attention, embedding etc. 
```
📦onmt
 ┣ 📂bin
 ┣ 📂decoders
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜decoder.py
 ┃ ┣ 📜ensemble.py
 ┃ ┗ 📜ltransformer.py      # Lorentz decoder
 ┣ 📂encoders
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜encoder.py
 ┃ ┗ 📜ltransformer.py      # Lorentz encoder
 ┣ 📂inputters
 ┣ 📂manifolds              # Lorentz manifold related
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜lmath.py
 ┃ ┣ 📜lorentz.py
 ┃ ┗ 📜utils.py
 ┣ 📂models
 ┣ 📂modules
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜copy_generator.py
 ┃ ┣ 📜hyper_nets.py        # Lorentz components, including linear and positional feed-forward
 ┃ ┣ 📜lembedding.py        # Lorentz embedding
 ┃ ┣ 📜lmulti_headed_attn.py    # Lorentz attention
 ┃ ┣ 📜source_noise.py
 ┃ ┗ 📜util_class.py
 ┣ 📂translate
 ┣ 📂utils
 ┣ 📜__init__.py
 ┣ 📜model_builder.py
 ┣ 📜opts.py
 ┣ 📜train_single.py
 ┗ 📜trainer.py
 ```

## 1. Usage
Take IWSLT'14 for example. 

1. Download and tokenize the dataset.

```bash
cd data/iwslt14/
bash prepare-iwslt14.sh
```

2. Preprocess the dataset.

```bash
cd ../../
bash preprocess.iwslt14.sh
```

1. Train the model. You can modify the parameters in `run.iwslt.64.sh`, then run the following command.
```bash
bash run.iwslt.64.sh
```

4. Evaluate the model.
```bash
bash eval_iwslt.sh ${beam_size} ${gpu_id} ${model_path}
```
e.g.,
```bash
bash eval_iwslt.sh 4 0 ./model/iwslt/model_step_40000.pt
```