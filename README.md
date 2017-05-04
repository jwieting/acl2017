# acl2017

Code to train models from "Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings".

The code is written in python and requires numpy, scipy, theano, and the lasagne libraries.

To get started, run setup.sh to download a trained Gated Recurrent Averaging Network (GRAN) model and required files such as training data and evaluation data. There is a demo script that takes the model that you would like to train as a command line argument (check the script to see available choices). Check main/train.py for command line options.

If you use our code for your work please cite:

@inproceedings{wieting-17-recurrent,
        author = {John Wieting and Kevin Gimpel},
        title = {Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings},
        booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
        year = {2017},
}
