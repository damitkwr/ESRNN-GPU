# Fast ES-RNN: A GPU Implementation of the ES-RNN Algorithm

A GPU-enabled version of the [hybrid ES-RNN model](https://eng.uber.com/m4-forecasting-competition/) by Slawek et al that won the M4 time-series forecasting competition by a large margin. The details of our implementation and the results are discussed in detail on this [paper](https://arxiv.org/abs/1907.03329)

## Getting Started

### Prerequisites

```
Python (3.5+)
Tensorflow (1.12+ to 1.14)
PyTorch (0.4.1)
Zalando Research's Dilated RNN
```

### Dataset

Please download the M4 competition dataset directly from [here](https://github.com/M4Competition/M4-methods/tree/master/Dataset) and put the files in the data directory.

### Running the algorithm

Either use an IDE such as PyCharm or make sure to add the es\_rnn folder to your PYTHON PATH before running the [main.py](es_rnn/main.py) in the es\_rnn folder. You can change the configurations of the algorithm in the [config.py](es_rnn/config.py) file.

## Built With

* [Python](https://www.python.org) - The *data science* language ;)
* [PyTorch](https://www.pytorch.org/) - The dynamic framework for computation


## Authors

* **Andrew Redd** - [aredd-cmu](https://github.com/aredd-cmu)
* **Kaung Khin** - [damitkwr](https://github.com/damitkwr)
* **Aldo Marini** - [catapulta](https://github.com/catapulta)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Thank you to the original author of the algorithm Smyl Slawek [slaweks17](https://github.com/slaweks17) for advice and for creating this amazing algorithm
* Zalando Research [zalandoresearch](https://www.github.com/zalandoresearch) for their implementation of Dilated RNN

## Citation

If you choose to use our implementation in your work please cite us as:

```
@article{ReddKhinMarini,
       author = {{Redd}, Andrew and {Khin}, Kaung and {Marini}, Aldo},
        title = "{Fast ES-RNN: A GPU Implementation of the ES-RNN Algorithm}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Jul",
          eid = {arXiv:1907.03329},
        pages = {arXiv:1907.03329},
archivePrefix = {arXiv},
       eprint = {1907.03329},
 primaryClass = {cs.LG}
}
```


#
