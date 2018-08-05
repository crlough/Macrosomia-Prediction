#Get the data into R
afecciones <- read.csv("AFECCIONES.csv")
egreso <- read.csv("EGRESO.csv")
defunc <- read.csv("DEFUNC.csv")
obstet <- read.csv("OBSTET.csv")
productos <- read.csv("PRODUCTOS.csv")
procedimientos <- read.csv("PROCEDIMIENTOS.csv")

#Necessary Packages to get between R and SQL
install.packages(c("dbplyr", "RSQLite", "dplyr"))
library(dbplyr)
library(RSQLite)
library(dplyr)


my_db_file <- "portal-database.sqlite"
my_db <- src_sqlite(my_db_file, create = TRUE)
copy_to(my_db, afecciones)
copy_to(my_db, egreso)
copy_to(my_db, defunc)
copy_to(my_db, obstet)
copy_to(my_db, productos)
copy_to(my_db, procedimientos)

#my_db
#database <- DBI::dbConnect(RSQLite::SQLite(), "my_db")
a <- tbl(my_db, "afecciones")
e <- tbl(my_db, "egreso")
d <- tbl(my_db, "defunc")
o <- tbl(my_db, "obstet")
p <- tbl(my_db, "productos")
pp <- tbl(my_db, "procedimientos")

Common.ID <- inner_join(o, e) %>%
  inner_join(pp) %>%
  inner_join(p) %>%
  collect()

#### MEETING ON 5/29, ADDITIONAL OMITTED VARIABLES
#Getting rid of chosen omitted variables
Common.ID <- Common.ID[-c(1,2,4,5,12,13,14,16,17,18,19,21,30,31,32,34,35,36,37,38,8,33,43,45,49,58,66,68,22,57,61,29
                          ,39,40,41,42,44,46,47,48,50,51,52,53,59,60,62,64,65,67,69)] #FROM 'VARIABLE TYPES.CSV'

Common.ID[,19] <- Common.ID$PESOPROD
Common.ID$V19[Common.ID$V19 <= 4000] <- 0 
Common.ID$V19[Common.ID$V19 > 4000] <- 1 

Common.ID$PESO[Common.ID$PESO==999] <- NA
Common.ID$PARTOS[Common.ID$PARTOS==99] <- NA
Common.ID$GESTAC[Common.ID$GESTAC==99] <- NA
Common.ID$TALLA[Common.ID$TALLA==999] <- NA
Common.ID$ENTIDAD[Common.ID$ENTIDAD==99] <- NA
Common.ID$MUNIC[Common.ID$MUNIC==999] <- NA
Common.ID$LOC[Common.ID$LOC==9999] <- NA

update_Common.ID <- na.omit(Common.ID)
Common.ID <- update_Common.ID



#####################################################

Common.ID$TIPNACI <- as.factor(Common.ID$TIPNACI)
Common.ID$PLANFAM <- as.factor(Common.ID$PLANFAM)
Common.ID$CLUES <- as.factor(Common.ID$CLUES)
Common.ID$DERHAB <- as.factor(Common.ID$DERHAB)
Common.ID$ENTIDAD <- as.factor(Common.ID$ENTIDAD)
Common.ID$MUNIC <- as.factor(Common.ID$MUNIC)
Common.ID$LOC <- as.factor(Common.ID$LOC)
Common.ID$MES_ESTADISTICO <- as.factor(Common.ID$MES_ESTADISTICO)
Common.ID$NUMPROMED <- as.factor(Common.ID$NUMPROMED)
Common.ID$PROMED <- as.factor(Common.ID$PROMED)
Common.ID$V19 <- as.factor(Common.ID$V19)

#
#
install.packages(glmnet)
library(glmnet)
install.packages(Matrix)
library(Matrix)
#
#### CLEANED DATA NEEDS TO BE LASSO'D BUT THE SET IS ENORMOUS, SO CREATE A TRAINING SET
train <- sample(1:nrow(Common.ID), 50000)
training <- Common.ID[train,]
for (i in 1:4) {
 toadd <- training[training$V19 %in% 1,] #up-sampling
 training <- rbind(training,toadd)
}
test <- Common.ID[-train,]

xfactors <- sparse.model.matrix(V19 ~ PLANFAM + MUNIC +
                                  DERHAB + 
                                  #+ CLUES + PROMED + LOC + ENTIDAD +
                                  MES_ESTADISTICO + NUMPROMED ,data=training)[, -1]
yfactors <- sparse.model.matrix(V19 ~ PLANFAM + MUNIC +
                                  DERHAB +
                                  #+ CLUES + PROMED + LOC + ENTIDAD +
                                  MES_ESTADISTICO + NUMPROMED ,data=test)[, -1]
m2 <- Matrix(training$TALLA, sparse = TRUE)
m3 <- Matrix(training$GESTAC, sparse = TRUE)
m4 <- Matrix(test$TALLA, sparse = TRUE)
m5 <- Matrix(test$GESTAC, sparse = TRUE)
x <- cBind(m2, m3, xfactors)
y <- cBind(m4, m5, yfactors)

########GLMNET
fit1 <- cv.glmnet(x, y=as.factor(training$V19), alpha=1, family="binomial", nfolds=5)
plot(fit1)
glmmod <- glmnet(x, y=as.factor(training$V19), alpha=1, family="binomial", standardize = TRUE)
plot(glmmod, xvar="lambda") # Plot variable coefficients vs. shrinkage parameter lambda.
model <- predict(glmmod, y, type= "response", s=fit1$lambda.min)
fit1$lambda.min #2.98*(10^-5)


model[model > .5] <- 1
model[model <= .5] <-0
model <- as.data.frame(model)
model$`1` <- as.factor(model$`1`)
training$V19 <- as.integer(training$V19) - 1
test$V19 <- as.factor(test$V19)
install.packages("caret")
library(caret)
table(model$`1`, test$V19 ) #12.6% error-rate
#Sensitivity 1691966/(1691966+176216) = .9
#Comparatively low specificity = 

myCoefs <- coef(fit1, s="lambda.min");
myCoefs[which(myCoefs != 0 ) ] 
myCoefs@Dimnames[[1]][which(myCoefs != 0 ) ]
myResults <- data.frame(
  features = myCoefs@Dimnames[[1]][ which(myCoefs != 0 ) ], #intercept included
  coefs    = myCoefs              [ which(myCoefs != 0 ) ]  #intercept included
)
myResults

#########GLM
#logitmodel <- glm(V19 ~., family = "binomial", data = training)
#summary(logitmodel)
#logitpred <- predict(logitmodel, test)
#summary(logitpred)
#logitpred[logitpred < .5] <- 0
#logitpred[logitpred >= .5] <- 1
#table(logitpred, test$V19) 

########## XGBoost
install.packages("xgboost")
library(xgboost)
bstSparse <- xgboost(data = x, label = training$V19, nround = 2, objective = "binary:logistic")
pred <- predict(bstSparse, y)
prediction <- as.numeric(pred > 0.5)
err <- mean(as.numeric(pred > 0.5) != test$V19)
print(paste("test-error=", err)) #"test-error= 0.128026970467018"
table(prediction, test$V19 )
