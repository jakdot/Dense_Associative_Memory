# Dense_Associative_Memory

An example of Dense Associative Memory training with a backpropagation algorithm on MNIST. Based on the paper [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164) by Dmitry Krotov and John Hopfield. If you want to learn more about Dense Associative Memories, check out a [NIPS 2016 talk](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Dense-Associative-Memory-for-Pattern-Recognition) or a [research seminar](https://www.youtube.com/watch?v=lvuAU_3t134). 

## Getting started

install jupyter notebook and numpy, scipy, matplotlib.

```bash
> jupyter notebook
```
run `Dense_Associative_Memory_training.ipynb` and observe the weights together with the errors on the training and the test sets.

## Author and License
(c) 2016 Dmitry Krotov
-- Apache 2.0 License

jakdot: I added comments to the ipynb file and added a few tests that should help understand the code. The file further testing describes some extra ways of exploring Hopfield networks. The python files show variation in training of Hopfield networks. The folder images show how training proceeded for three different cases - when n was small (n=3), when it was standard (n=30) and when fewer memories were stored (16 instead of 100). The pkl file stores a trained network.
