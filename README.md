# INTRO

Principal Component Analysis (PCA) is a widely used tool to easily find
patterns of similarity in a multivariate dataset.

The goal of PCA is to reduce the dimensionality of the data by computing principal
components that are linear combinations of the original variables, and that accounts
for as much of the variability in the data as possible.
In this way, the information contained in the original dataset is mapped to a
lower n-dimensional space where it can be more easily processed.

For example, computing the first 2 principal components from a N-dimensional
dataset is equivalent of finding the two-dimensional plane through the N dimensions
in which the data is most spread out.
So a dataset containing subclasses with dissimilarities with respect to the original
variables will be mapped to a 2-dimensional space showing clusters for each subclass
when plotted out in a two-dimensional scatter plot.

Once the coefficients of the principal components are knowns, any new data point
represented by the same variables of the original dataset can be mapped to the
lower dimensional representation.

# USAGE

A dataset can be viewed as a matrix X of N_s samples and N_c columns, each column
representing the distribution of population variables, and a vector y representing
class information.

The tool allows to perform Principal Component Analysis and plot the transformed
data onto a scatter plot of the first two Principal Components (PC^1, PC^2). 95%
confidence ellipsis for each class are drawn.

Once the model is trained, a new sample can be mapped to the lower dimensional
representation to overlay the scatter plot.

# EXAMPLE

PCA is used in forensic life-science to verify the authenticity of a wine sample,
once information on its chemical composition is known.
Once a PCA model is trained with a dataset containing chemical analysis on
different wine samples and relative origin information, it can be used to
visually verify the origin of new samples, checking whether they are projected
within the confidence ellipsis of their relative cluster.

In the example, data from http://archive.ics.uci.edu/ml/datasets/Wine contain
the results of a chemical analysis of wines grown
in the same region in Italy but derived from three different cultivars.
The analysis determined the quantities of 13 constituents found in each of the
three types of wines.

After normalisation, we pick a wine sample at random from the original dataset,
and compute PCA with the remaining data.
Finally we use the trained model to verify the origin of the sample.
