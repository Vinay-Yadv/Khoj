bank <- droplevels(bank[!grepl("^\\s*$", bank$job),,drop=FALSE] )
str(bank)
bankTest <- droplevels(bankTest[!grepl("^\\s*$", bankTest$job),,drop=FALSE] )

classif.task <- makeClassifTask(id = "bank", data = bank, target = "term.deposit",positive= "yes")
classif.task

bank$euribor3m <- NULL
bank$emp.var.rate <- NULL

bankTest$euribor3m <- NULL
bankTest$emp.var.rate <- NULL

fv <- generateFilterValuesData(classif.task)
 
plotFilterValues(fv) + coord_flip()

filteredTask2 <- filterFeatures(classif.task, fval = fv, perc = 0.6)

lrn <- makeFilterWrapper(learner = "classif.kknn", fw.method = "chi.squared")

ps <- makeParamSet(makeDiscreteParam("fw.perc", values = seq(0.2, 0.5, 0.05)))
rdesc <- makeResampleDesc("CV", iters = 3)
res <- tuneParams(lrn, task = classif.task,
                  resampling = rdesc,
                  par.set = ps,
                  control = makeTuneControlGrid())

fusedLrn <- makeFilterWrapper(learner = "classif.kknn",
                              fw.method = "chi.squared",
                              fw.perc = res$x$fw.perc)
mod <- train(fusedLrn, classif.task)
mod

getFilteredFeatures(mod)

randomCtrl <- makeFeatSelControlRandom(maxit = 10L)
rdesc <- makeResampleDesc("CV", iters = 3)
sfeats <- selectFeatures(learner = "classif.kknn", task = classif.task,
                         resampling = rdesc, control = randomCtrl, show.info = FALSE)

sfeats$x
sfeats$y


learner1 <- makeLearner(cl = "classif.naiveBayes",
                        predict.type = "prob")
modn <- train( learner1, classif.task)
predn <- predict(modn, testTask)
performance(predn)

ms= list(tp, tn, fp, fn)
performance(predn, measures = ms )

calculateConfusionMatrix(predn)
predn$threshold


#################3################
lrn3 <- makeLearner(cl="classif.randomForest", predict.type = 'prob')

mod3 <- mlr::train(lrn3, train2)

predrf <- predict(mod3, test2)

performance(predrf)
ms= list(tp, tn, fp, fn)
performance(predrf, measures = ms )

calculateConfusionMatrix(predrf)

measure <- list(mmce, tpr, fnr)
bmr <- benchmark(lrn3, trainTask, rdesc, measure)

########################################################
bank2<- bank 
bank2$term.deposit <- as.numeric(bank2$term.deposit)


bankTest2<- bankTest 
bankTest2$term.deposit <- as.numeric(bankTest2$term.deposit)

dmy <- dummyVars(" ~.", data = bank2, fullRank = T, levelsOnly = FALSE)

trsf <- data.frame(predict(dmy, newdata = bank2))
trsf
trsf$term.deposit <- as.factor(trsf$term.deposit)

dmy2 <- dummyVars(" ~ .", data = bankTest2, fullRank = T,levelsOnly = FALSE)

trsf2 <- data.frame(predict(dmy2, newdata = bankTest2))
trsf2$term.deposit <- as.factor(trsf2$term.deposit)

train2 <- makeClassifTask( data = trsf, target = "term.deposit")

test2 <- makeClassifTask( data = trsf2, target = "term.deposit")


lrn3 <- makeLearner(cl="classif.randomForest", predict.type = 'prob')

mod3 <- mlr::train(lrn3, train2)

predrf <- predict(mod3, test2)

performance(predrf)
ms= list(tp, tn, fp, fn)
performance(predrf, measures = ms )

calculateConfusionMatrix(predrf)

measure <- list(mmce, tpr, fnr)
bmr <- benchmark(lrn3, trainTask, rdesc, measure)

performance(pred_on_test,  measures = auc)
performance(pred_on_test_spsa_tree_tuned, measures = auc)
performance(pred_on_test_spsa_tree, measures = auc)
```
