# Apriori

# Data Preprocessing
# install.packages('arules')
library(arules) 
# 'arules' library does not take csv file as input, rather it takes 'sparse matrix'
# Sparse Matrix is a matrix with very few non-zero elements
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
# header = FALSE tells the R that the 1st line of dataset does not contain the titles of the columns
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])