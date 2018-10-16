# Introduction

In this project, we explore the differences between Principal Component Analysis (PCA), kernelized PCA and Gaussian Process latent variable models (gp-lvm). All three methods belong to the world of unsupervised learning. In particular, these methods aim to find a low dimensional representation or code for the data.

_This readme does not aim to explain these methods, I briefly introduce them and refer to other machine learning books for details_

For example, we might have 20 properties of a person, like its height, weight, shoe size, age and in many more properties. We would like to find two or three numbers that represent the same information in all these properties. This reduction in dimensions also allows for visualisations. Imagine we have twenty properties of a person. We can't plot twenty dimensions. However, we can find a two or three dimensional representation and plot those low dimensional representations.

# What is a latent variable model?
We can explain machine learning model by the assumptions on the process that generate the data. For latent variable model, we assume that we draw a random vector, <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/>, from an assumed distribution. Then we convert this vector <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/> into another vector, <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=14.102549999999994pt/>. The latent variable model is defined by the assumed distribution on <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/> and the assumed conversion to <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=14.102549999999994pt/>. In this case, we name the vector <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/> the latent variable and vector <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=14.102549999999994pt/> the observed vector. 

# How do we get from PCA to kPCA to GP LVM?
We can consider the three methods an extension of one another. We start with explaining PCA, extend that explanation to kPCA and finally arrive at GP LVM. 

In principal component analysis, we assum that the prior distribution over <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/> is a Gaussian distribution. So every time we need a new realisation of <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/>, we draw a random vector from a <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/946450d4c85f0a822df0057745111cfc.svg?invert_in_darkmode" align=middle width=52.263090000000005pt height=24.56552999999997pt/> distribution. Our second assumption reads that the conversion from <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.336460000000002pt height=14.102549999999994pt/> to <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=14.102549999999994pt/> is a linear mapping, defined by matrix <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.74179pt height=22.381919999999983pt/>. We then observe a noise version of <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=14.102549999999994pt/>, as we assume that some noise of another Gaussian is added. In Murphy's book, chapter twelve, you can read how these assumptions lead to the algorithm that performs inference.

In the algorithm, you will read that the principal components follow from a _singular value decomposition_ of the sample covariance matrix. We can rewrite these computations in terms of the inner product of observed vectors. Now this might ring a bell to you. As soon as some algorithm can be written as function of inner products, we can abstract this inner product into a kernel computation. A kernel abstract away the similarity of two vectors. It inputs two vectors <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/8c76e0c69c5596634f9abb693bbf9438.svg?invert_in_darkmode" align=middle width=17.548410000000004pt height=21.10812pt/> and <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/1533fefb8348ed2119c7920bf5d7a8a5.svg?invert_in_darkmode" align=middle width=17.548410000000004pt height=21.10812pt/> and outputs a real number, representing similarity. The simplest case implements the linear kernel: <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/09a0f463995e9c2eef43d9fe84bc8d6e.svg?invert_in_darkmode" align=middle width=131.48899500000002pt height=27.598230000000008pt/>.

This abstraction leads us to kernel PCA. I personally find it hard to visualize the data generation process. I hope to be able to include the explanation here some time in the future.

Moving to the GP LVM, we switch our optimization problem. In PCA and kernel PCA, we optimize the parameters of the conversion from latent space to observed space. For example, we find the vectors of the principal components for PCA. Then we calculate the latent representations using these parameters. You might question why we not directly find the latent representations and forget about the conversion. GP implements exactly this reasoning: instead of optimizing over parameters, we intergrate out the parameters and optimize directly over the latent representations. 

Unfortunately, this now optimization problem does not have such elegant form as PCA does. For a maximum likelihood estimate of the GP LVM, we do an iterative optimization. In this project, we use the implementation of the GP LVM in [GPy](https://gpy.readthedocs.io/en/deploy/). 

# Data
We construct a synthetic data set to explore these three methods. In our case, we want to explore non linear manifolds. Therefore, we synthesize data from a Swiss roll: <img src="https://github.com/RobRomijnders/squashing/blob/master/svgs/16432370dbc39f5de7364a1c2974403d.svg?invert_in_darkmode" align=middle width=257.19919500000003pt height=27.720329999999983pt/> and add data to all three axes. 

# Results

## PCA
The upper left image results from a PCA. The x and y axis have larger spread than the z axis. Therefore we expect the x- and y-axis to be the first two principal components. The low dimensional embedding is the swiss roll without z axis

## kPCA
Kernel PCA is inherently more sensitive to local similarities. The upper right plot displays the results of kPCA. We see that kPCA disentangles the yellow and purple sling.

## GP LVM
At the heart of GP LVM lies an optimization problem. Therefore, the results is in the hands of the optimizer and different settings of the optimizer lead to different results. GP LVM is a kernel method as well, caring more about local similarities. We observe in the lower left diagram that GP LVM also disentangles the two slings of the Swiss roll. However, the representations are not as contiguous as the resulting embeddings of the kPCA.

![image](https://github.com/RobRomijnders/kpca_gplvm/blob/master/doc/comparison_equal_variance.png?raw=true)


# Further reading

  * [Murphy's book, chapter twelve on latent variable models](https://mitpress.mit.edu/books/machine-learning-1)
  * [GPy - GPy is a Gaussian Process (GP) framework written in Python, from the Sheffield machine learning group.](https://gpy.readthedocs.io/en/deploy/)
  * [Original NIPS paper on GP LVM](https://papers.nips.cc/paper/2540-gaussian-process-latent-variable-models-for-visualisation-of-high-dimensional-data.pdf)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com