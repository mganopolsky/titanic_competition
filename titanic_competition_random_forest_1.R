# This R environment comes with many helpful analytics packages installed
# It is defined by the kaggle/rstats Docker image: https://github.com/kaggle/docker-rstats
# For example, here's a helpful package to load

library(tidyverse) # metapackage of all tidyverse packages
library(randomForest)
library(caret)
library(party)
library(ggplot2)
library(dplyr)
library(GGally)
library(rpart)
library(rpart.plot)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
dir_path <- "../input/titanic/"
#list.files(path = dir_path)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_set <- read.csv(paste(dir_path, "train.csv", sep=""))
#head(train_set)
test_set <- read.csv(paste(dir_path, "test.csv", sep=""))
#head(test_set)

train_set <- train_set %>% mutate_at( vars(Sex), as.factor) %>% mutate_at(vars(Survived), as.factor)
test_set <- test_set %>% mutate_at( vars(Sex), as.factor) 


women <- train_set %>% filter(Sex == "female")
men <- train_set %>% filter(Sex == "male")

percent_women_survived <- women %>% filter(Survived == 1) %>% summarize(n()) / nrow(women)
percent_men_survived <- men %>% filter(Survived == 1) %>% summarize(n()) / nrow(men)
percent_women_survived
percent_men_survived

#no missing passengerIds
length(unique(train_set$PassengerId)) == nrow(train_set)

train_set_count <- nrow(train_set)


#are there missing ages?
avg_age_train <- train_set %>% group_by(Pclass, Sex) %>% summarize(avg_age = mean(Age,  na.rm = TRUE))  %>% as.data.frame()
avg_age_test <- test_set %>% group_by(Pclass, Sex) %>% summarize(avg_age = mean(Age,  na.rm = TRUE))  %>% as.data.frame()

avg_fare_pclass_train <- train_set %>% group_by(Pclass) %>% summarize(avg_fare = mean(Fare,  na.rm = TRUE)) %>% as.data.frame()
avg_fare_pclass_test <- test_set %>% group_by(Pclass) %>% summarize(avg_fare = mean(Fare,  na.rm = TRUE))  %>% as.data.frame()

merge_cols <- c("Pclass", "Sex")

train_set <- inner_join(train_set, avg_age_train, by = merge_cols)
test_set <- inner_join(test_set, avg_age_test, by = merge_cols)

train_set <- inner_join(train_set, avg_fare_pclass_train, by = "Pclass")
test_set <- inner_join(test_set, avg_fare_pclass_test, by = "Pclass")

#update Age is it's NA based on AVG age for that PClass and Sex
train_set <- train_set %>% mutate(Age = ifelse(is.na(Age) , avg_age, Age)) %>% mutate(Child = ifelse(Age < 18, 1, 0))
test_set <- test_set %>% mutate(Age = ifelse(is.na(Age) , avg_age, Age)) %>% mutate(Child = ifelse(Age < 18, 1, 0))




#update Fare is it's NA based on AVG age for that PClass and Sex
train_set <- train_set %>% mutate(Fare = ifelse(Fare == 0 | is.na(Fare) , avg_fare, Fare))
test_set <- test_set %>% mutate(Fare = ifelse(Fare == 0  | is.na(Fare), avg_fare, Fare))

#add 'Family_Size'field as combo of sibblings and parent/child count
train_set <- train_set %>% mutate(Family_Size = SibSp + Parch )
test_set <- test_set %>% mutate(Family_Size = SibSp + Parch )


#add 'Age*Class'field as product of Age * Class
train_set <- train_set %>% mutate(Age_Class = Age + Pclass )
test_set <- test_set %>% mutate(Age_Class = Age + Pclass )

#try a random forest machine learning model 
y <- train_set$Survived

features <- c("Pclass", "Sex", "SibSp", "Parch")
X <- train_set %>% select(features)

set.seed(1)

fit_1 <- randomForest(Survived ~ Child + Pclass + Sex + Age + SibSp + Parch + Fare +
                      Family_Size,
                    data=train_set, 
                    importance=TRUE, 
                    ntree=2000)

#varImpPlot(fit)

#this is the first set of predictions using random forest
prediction_1 <- predict(fit_1, test_set)

#
set.seed(450)


fit_2 <- cforest(Survived ~ Child + Pclass + Sex + Age + SibSp + Parch + Fare +
                 Family_Size,
               data = train_set, 
               controls=cforest_unbiased(ntree=2000, mtry=3))

#this is the second set of predictions using random forest. different library
prediction_2 <- predict(object = fit_2, newdata = test_set, OOB=TRUE, type = "response")


#set.seed(1)
#model_3 <- glm( Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +Family_Size + Child , data = train_set, family = binomial)
#summary(model_3)$coef

#probabilities <- model_3 %>% predict(test_set, type = "response")
#head(probabilities)

#predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
#head(predicted.classes)



#model_dt<- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +Family_Size  + Child,data=train_set, method="class")
#rpart.plot(model_dt)


#pred_test_eval <- predict(model_dt, test_set, type = "class")

submit <- data.frame(PassengerId = test_set$PassengerId, Survived = prediction_2)

write.csv(submit,'submission_l_1.csv', row.names = FALSE)
