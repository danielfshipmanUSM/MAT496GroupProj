# MAT496GroupProj
Various python scripts to assist data manipulation for the MAT 496 Datascience group project

The idea is to keep track of changes to our scripts. 

Currently the DataCleanup.py takes the true and false datasets found in Data folder. Make sure to unzip 
it iterates through all words, then parses out just letters and capitals. This is to avoid differences between
"this", "This" and "This.". 

afterwords it stores all unique words as the columns of a dataset, these will be our predictors.

It then associates a count within all specific data entries of each word. In other words, it stores how often a unique word is used for each article. 
The goal is to use these counts to determine some kind of classifier. However the current problem is that there are about 60,000 predictors and a sample of only 20,000

To mitigate this problem, we attempt various kinds of feature selection. 

Due to the high dimensionality of the predictors, however, some algorithms are entirely infeasible. For instance "Best subset selection" simply wont happen. 

TODO implement feature selection 
  - Try "Forward Step Selection"
