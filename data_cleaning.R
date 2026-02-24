
data <- read.csv("data/bankruptcy.csv")
str(data)
summary(data)
colSums(is.na(data))
sum(is.na(data))

# Convert target variable to factor
data$Bankrupt. <- as.factor(data$Bankrupt.)

# Check class balance
table(data$Bankrupt.)

# Check percentage
prop.table(table(data$Bankrupt.))
