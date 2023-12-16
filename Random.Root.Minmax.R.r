install.packages("randomForest") ; library(randomForest)
install.packages("xgboost") ; library(xgboost)
install.packages("lightgbm") ; library(lightgbm)
install.packages("pROC") ; library(pROC)
install.packages("caret") ; library(caret)
install.packages("adabag") ; library(adabag)
library(rpart) ; library(e1071) ; library(class) ; library(MASS) ; library(nnet)

setwd("C:/Users/0206d/OneDrive/바탕 화면/학교/4학년 1학기/데마 팀플/")
dat = read.csv("data_max.csv", fileEncoding = "euc-kr")

b.rst.tab = matrix(NA, 3, 60)
rownames(b.rst.tab) = c("acc", "kpa", "auc")
colnames(b.rst.tab) = c("M_LOG", "M_DT", "M_RF", "M_RF_1", "M_RF_2", "M_RF_3", "M_RF_4" ,"M_NB", "M_SVM",
                      "M_KNN","M_KNN_1", "M_KNN_2", "M_KNN_3", "M_KNN_4", "M_LDA", "M_NN_1", "M_NN_2", "M_NN_3", "M_NN_4",
                      "M_AB", "M_XGB", "M_XGB_1", "M_XGB_2", "M_XGB_3", "M_XGB_4", "M_LGB", "M_LGB_1", "M_LGB_2", "M_LGB_3", "M_LGB_4",
                      "OR_LOG", "OR_DT", "OR_RF", "OR_RF_1", "OR_RF_2", "OR_RF_3", "OR_RF_4" ,"OR_NB", "OR_SVM",
                      "OR_KNN","OR_KNN_1", "OR_KNN_2", "OR_KNN_3", "OR_KNN_4", "OR_LDA", "OR_NN_1", "OR_NN_2", "OR_NN_3", "OR_NN_4",
                      "OR_AB", "OR_XGB", "OR_XGB_1", "OR_XGB_2", "OR_XGB_3", "OR_XGB_4", "OR_LGB", "OR_LGB_1", "OR_LGB_2", "OR_LGB_3", "OR_LGB_4")

n=30
acc.mat = kpa.mat = auc.mat = matrix(NA, n, 60)

sqrt.fun = function(data){
  a.vec = c(5,7,8,9,10,11,12)
  for (i in a.vec){data[,i] = sqrt(data[,i])}
  return(data)
}

scaling.minmax = function(data1,data2){
  newdata = data1[,-c(1,ncol(data1))] 
  olddata = data2[,-c(1,ncol(data2))]
  for(i in 1:ncol(newdata)){
    newdata[i] = (newdata[,i]-min(olddata[,i])) / (max(olddata[,i])-min(olddata[,i]))
  }
  data.scaling.minmax = cbind(data1[,1], newdata, data1[,ncol(data1)])
  colnames(data.scaling.minmax)[c(1,ncol(data1))] = c('강수여부', 'target')
  data.scaling.minmax$target = as.factor(data.scaling.minmax$target)
  return(data.scaling.minmax)
}

system.time(for(iter in 1:n){
  dat$target = as.factor(dat$target)
  set = sample(1:nrow(dat), 0.5*nrow(dat))
  train = dat[set,]
  test = dat[-set,]
  
  nset = sample(1:nrow(train), 2000)
  osam = sqrt.fun(train[nset,])
  val = sqrt.fun(train[-nset,])
  sam = scaling.minmax(osam,osam)
  val = scaling.minmax(val,osam)
  
  #-----------------------------------------logistic
  # logistic 모델 학습
  log_model = multinom(target ~ ., data = sam)
  # 예측 수행
  log_pred = predict(log_model, val[,-ncol(dat)], type = "class")
  # 예측 결과 평가
  acc.mat[iter,1] = confusionMatrix(log_pred, val$target)$overall[1]
  kpa.mat[iter,1] = confusionMatrix(log_pred, val$target)$overall[2]
  auc.mat[iter,1] = multiclass.roc(val$target, as.numeric(log_pred)-1)$auc[1]
  
  #-----------------------------------------Decision Tree
  # Decision Tree 모델 학습
  dt_model = rpart(target ~ ., data = sam, method = "class")
  # 예측 수행
  dt_pred = predict(dt_model, val[,-ncol(dat)], type = "class")
  #예측 결과 평가
  acc.mat[iter,2] = confusionMatrix(dt_pred, val$target)$overall[1]
  kpa.mat[iter,2] = confusionMatrix(dt_pred, val$target)$overall[2]
  auc.mat[iter,2] = multiclass.roc(val$target, as.numeric(dt_pred)-1)$auc[1]
  
  #-----------------------------------------randomforest
  # randomforest 모델 학습
  rf_model = randomForest(target ~ ., data = sam, ntree = 100)
  # 예측 수행
  rf_pred = predict(rf_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,3] = confusionMatrix(rf_pred, val$target)$overall[1]
  kpa.mat[iter,3] = confusionMatrix(rf_pred, val$target)$overall[2]
  auc.mat[iter,3] = multiclass.roc(val$target, as.numeric(rf_pred)-1)$auc[1]
  
  # randomforest 모델 학습
  rf_model = randomForest(target ~ ., data = sam, ntree = 100, mtry = 3)
  # 예측 수행
  rf_pred = predict(rf_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,4] = confusionMatrix(rf_pred, val$target)$overall[1]
  kpa.mat[iter,4] = confusionMatrix(rf_pred, val$target)$overall[2]
  auc.mat[iter,4] = multiclass.roc(val$target, as.numeric(rf_pred)-1)$auc[1]
  
  # randomforest 모델 학습
  rf_model = randomForest(target ~ ., data = sam, ntree = 100, mtry = 4)
  # 예측 수행
  rf_pred = predict(rf_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,5] = confusionMatrix(rf_pred, val$target)$overall[1]
  kpa.mat[iter,5] = confusionMatrix(rf_pred, val$target)$overall[2]
  auc.mat[iter,5] = multiclass.roc(val$target, as.numeric(rf_pred)-1)$auc[1]
  
  # randomforest 모델 학습
  rf_model = randomForest(target ~ ., data = sam, ntree = 100, mtry = 5)
  # 예측 수행
  rf_pred = predict(rf_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,6] = confusionMatrix(rf_pred, val$target)$overall[1]
  kpa.mat[iter,6] = confusionMatrix(rf_pred, val$target)$overall[2]
  auc.mat[iter,6] = multiclass.roc(val$target, as.numeric(rf_pred)-1)$auc[1]
  
  # randomforest 모델 학습
  rf_model = randomForest(target ~ ., data = sam, ntree = 100, mtry = 6)
  # 예측 수행
  rf_pred = predict(rf_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,7] = confusionMatrix(rf_pred, val$target)$overall[1]
  kpa.mat[iter,7] = confusionMatrix(rf_pred, val$target)$overall[2]
  auc.mat[iter,7] = multiclass.roc(val$target, as.numeric(rf_pred)-1)$auc[1]
  
  #-----------------------------------------Naive-Bayesian
  #Naive-Bayesian 모델 학습
  nb_model = naiveBayes(sam[,-ncol(dat)], sam$target)
  # 예측 수행
  nb_pred = predict(nb_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,8] = confusionMatrix(nb_pred, val$target)$overall[1]
  kpa.mat[iter,8] = confusionMatrix(nb_pred, val$target)$overall[2]
  auc.mat[iter,8] = multiclass.roc(val$target, as.numeric(nb_pred)-1)$auc[1]
  
  #-----------------------------------------SVM
  #SVM 모델 학습
  svm_model = svm(target~ ., data = sam)
  # 예측 수행
  svm_pred = predict(svm_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,9] = confusionMatrix(svm_pred, val$target)$overall[1]
  kpa.mat[iter,9] = confusionMatrix(svm_pred, val$target)$overall[2]
  auc.mat[iter,9] = multiclass.roc(val$target, as.numeric(svm_pred)-1)$auc[1]
  
  #-----------------------------------------KNN
  #KNN 모델 학습
  knn_model = knn(sam[,-ncol(dat)], val[,-ncol(dat)], cl = sam$target)
  # 예측 결과 평가
  acc.mat[iter,10] = confusionMatrix(knn_model, val$target)$overall[1]
  kpa.mat[iter,10] = confusionMatrix(knn_model, val$target)$overall[2]
  auc.mat[iter,10] = multiclass.roc(val$target, as.numeric(knn_model)-1)$auc[1]
  
  #KNN 모델 학습
  knn_model = knn(sam[,-ncol(dat)], val[,-ncol(dat)], cl = sam$target, k=3)
  # 예측 결과 평가
  acc.mat[iter,11] = confusionMatrix(knn_model, val$target)$overall[1]
  kpa.mat[iter,11] = confusionMatrix(knn_model, val$target)$overall[2]
  auc.mat[iter,11] = multiclass.roc(val$target, as.numeric(knn_model)-1)$auc[1]
  
  #KNN 모델 학습
  knn_model = knn(sam[,-ncol(dat)], val[,-ncol(dat)], cl = sam$target, k=5)
  # 예측 결과 평가
  acc.mat[iter,12] = confusionMatrix(knn_model, val$target)$overall[1]
  kpa.mat[iter,12] = confusionMatrix(knn_model, val$target)$overall[2]
  auc.mat[iter,12] = multiclass.roc(val$target, as.numeric(knn_model)-1)$auc[1]
  
  #KNN 모델 학습
  knn_model = knn(sam[,-ncol(dat)], val[,-ncol(dat)], cl = sam$target, k=10)
  # 예측 결과 평가
  acc.mat[iter,13] = confusionMatrix(knn_model, val$target)$overall[1]
  kpa.mat[iter,13] = confusionMatrix(knn_model, val$target)$overall[2]
  auc.mat[iter,13] = multiclass.roc(val$target, as.numeric(knn_model)-1)$auc[1]
  
  #KNN 모델 학습
  knn_model = knn(sam[,-ncol(dat)], val[,-ncol(dat)], cl = sam$target, k=15)
  # 예측 결과 평가
  acc.mat[iter,14] = confusionMatrix(knn_model, val$target)$overall[1]
  kpa.mat[iter,14] = confusionMatrix(knn_model, val$target)$overall[2]
  auc.mat[iter,14] = multiclass.roc(val$target, as.numeric(knn_model)-1)$auc[1]
  
  #-----------------------------------------LDA
  #LDA 모델 학습
  lda_model = lda(target ~ ., data = sam)
  # 예측 수행
  lda_pred = predict(lda_model, val[,-ncol(dat)])$class
  # 예측 결과 평가
  acc.mat[iter,15] = confusionMatrix(lda_pred, val$target)$overall[1]
  kpa.mat[iter,15] = confusionMatrix(lda_pred, val$target)$overall[2]
  auc.mat[iter,15] = multiclass.roc(val$target, as.numeric(lda_pred)-1)$auc[1]
  
  #-----------------------------------------Neural Networks
  #Neural Networks 모델 학습
  nn_model = nnet(target ~ ., data = sam, size = 3, maxit = 100)
  # 예측 수행
  nn_pred = as.numeric(predict(nn_model, val[,-ncol(dat)], type = "class"))
  # 예측 결과 평가
  acc.mat[iter,16] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,16] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,16] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  #Neural Networks 모델 학습
  nn_model = nnet(target ~ ., data = sam, size = 6, maxit = 100)
  # 예측 수행
  nn_pred = as.numeric(predict(nn_model, val[,-ncol(dat)], type = "class"))
  # 예측 결과 평가
  acc.mat[iter,17] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,17] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,17] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  nn_model = nnet(target ~ ., data = sam, size = 9, maxit = 100)
  # 예측 수행
  nn_pred = as.numeric(predict(nn_model, val[,-ncol(dat)], type = "class"))
  # 예측 결과 평가
  acc.mat[iter,18] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,18] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,18] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  nn_model = nnet(target ~ ., data = sam, size = 12, maxit = 100)
  # 예측 수행
  nn_pred = as.numeric(predict(nn_model, val[,-ncol(dat)], type = "class"))
  # 예측 결과 평가
  acc.mat[iter,19] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,19] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,19] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  #-----------------------------------------Adaboost
  # Adaboost 모델 학습
  ab_model = boosting(target ~ ., data = sam, mfinal = 100)
  # 예측 수행
  ab_pred = predict(ab_model, val[,-ncol(dat)])
  # 예측 결과 평가
  acc.mat[iter,20] = confusionMatrix(as.factor(ab_pred$class), val$target)$overall[1]
  kpa.mat[iter,20] = confusionMatrix(as.factor(ab_pred$class), val$target)$overall[2]
  auc.mat[iter,20] = multiclass.roc(val$target, as.numeric(as.factor(ab_pred$class))-1)$auc[1]
  
  #-----------------------------------------as.numeric
  sam$target = as.numeric(sam$target)-1
  
  #-----------------------------------------xgboost
  # XGBoost 모델 학습
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)))
  xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(val[,-ncol(dat)]))
  # 예측 결과 평가
  acc.mat[iter,21] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,21] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,21] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  # XGBoost 모델 학습
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)), max_depth = 3)
  xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(val[,-ncol(dat)]))
  # 예측 결과 평가
  acc.mat[iter,22] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,22] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,22] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  # XGBoost 모델 학습
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)), max_depth = 6)
  xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(val[,-ncol(dat)]))
  # 예측 결과 평가
  acc.mat[iter,23] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,23] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,23] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  # XGBoost 모델 학습
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)), max_depth = 9)
  xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(val[,-ncol(dat)]))
  # 예측 결과 평가
  acc.mat[iter,24] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,24] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,24] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  # XGBoost 모델 학습
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)), max_depth = 12)
  xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(val[,-ncol(dat)]))
  # 예측 결과 평가
  acc.mat[iter,25] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,25] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,25] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  #-----------------------------------------lightgbm
  # lightGBM 모델 학습
  lgb_train = lgb.Dataset(data = as.matrix(sam[, -ncol(dat)]), label = sam$target)
  lgb_params = list(objective = "multiclass", num_class = length(unique(dat$target)))
  lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
  # 예측 수행
  lgb_pred = max.col(predict(lgb_model,  as.matrix(val[, -ncol(dat)]), reshape=T))-1
  # 예측 결과 평가
  acc.mat[iter,26] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,26] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,26] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  # lightGBM 모델 학습
  lgb_train = lgb.Dataset(data = as.matrix(sam[, -ncol(dat)]), label = sam$target)
  lgb_params = list(objective = "multiclass", num_class = length(unique(dat$target)), max_depth = 3)
  lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
  # 예측 수행
  lgb_pred = max.col(predict(lgb_model,  as.matrix(val[, -ncol(dat)]), reshape=T))-1
  # 예측 결과 평가
  acc.mat[iter,27] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,27] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,27] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  # lightGBM 모델 학습
  lgb_train = lgb.Dataset(data = as.matrix(sam[, -ncol(dat)]), label = sam$target)
  lgb_params = list(objective = "multiclass", num_class = length(unique(dat$target)), max_depth = 6)
  lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
  # 예측 수행
  lgb_pred = max.col(predict(lgb_model,  as.matrix(val[, -ncol(dat)]), reshape=T))-1
  # 예측 결과 평가
  acc.mat[iter,28] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,28] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,28] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  lgb_train = lgb.Dataset(data = as.matrix(sam[, -ncol(dat)]), label = sam$target)
  lgb_params = list(objective = "multiclass", num_class = length(unique(dat$target)), max_depth = 9)
  lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
  # 예측 수행
  lgb_pred = max.col(predict(lgb_model,  as.matrix(val[, -ncol(dat)]), reshape=T))-1
  # 예측 결과 평가
  acc.mat[iter,29] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,29] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,29] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  lgb_train = lgb.Dataset(data = as.matrix(sam[, -ncol(dat)]), label = sam$target)
  lgb_params = list(objective = "multiclass", num_class = length(unique(dat$target)), max_depth = 12)
  lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
  # 예측 수행
  lgb_pred = max.col(predict(lgb_model,  as.matrix(val[, -ncol(dat)]), reshape=T))-1
  # 예측 결과 평가
  acc.mat[iter,30] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,30] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,30] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  #-----------------------------------------------------------------------------------------------------------ONE vs REST
  osam = sqrt.fun(train[nset,])
  val = sqrt.fun(train[-nset,])
  sam = scaling.minmax(osam,osam)
  val = scaling.minmax(val,osam)
  
  orsam = sam
  
  prob_pred = matrix(NA, nrow=nrow(train)-2000, ncol=length(unique(dat$target)))
  #-----------------------------------------logistic
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # logistic 모델 학습
    log_model =  multinom(target ~ ., data = orsam)
    # 예측 수행
    prob_pred[,iiter] = 1-predict(log_model, val[,-ncol(dat)], type = "probs")
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  log_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,31] = confusionMatrix(as.factor(log_pred), val$target)$overall[1]
  kpa.mat[iter,31] = confusionMatrix(as.factor(log_pred), val$target)$overall[2]
  auc.mat[iter,31] = multiclass.roc(val$target, log_pred)$auc[1]
  
  #-----------------------------------------Decision Tree
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # Decision Tree 모델 학습
    dt_model = rpart(target ~ ., data = orsam, method = "class")
    # 예측 수행
    prob_pred[,iiter] = predict(dt_model, val[,-ncol(dat)], type = "prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  dt_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,32] = confusionMatrix(as.factor(dt_pred), val$target)$overall[1]
  kpa.mat[iter,32] = confusionMatrix(as.factor(dt_pred), val$target)$overall[2]
  auc.mat[iter,32] = multiclass.roc(val$target, dt_pred)$auc[1]
  
  #-----------------------------------------randomforest
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target ~ ., data = orsam, ntree = 100)
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, val[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,33] = confusionMatrix(as.factor(rf_pred), val$target)$overall[1]
  kpa.mat[iter,33] = confusionMatrix(as.factor(rf_pred), val$target)$overall[2]
  auc.mat[iter,33] = multiclass.roc(val$target, rf_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target ~ ., data = orsam, ntree = 100, mtry = 3)
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, val[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,34] = confusionMatrix(as.factor(rf_pred), val$target)$overall[1]
  kpa.mat[iter,34] = confusionMatrix(as.factor(rf_pred), val$target)$overall[2]
  auc.mat[iter,34] = multiclass.roc(val$target, rf_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target ~ ., data = orsam, ntree = 100, mtry = 4)
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, val[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,35] = confusionMatrix(as.factor(rf_pred), val$target)$overall[1]
  kpa.mat[iter,35] = confusionMatrix(as.factor(rf_pred), val$target)$overall[2]
  auc.mat[iter,35] = multiclass.roc(val$target, rf_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target ~ ., data = orsam, ntree = 100, mtry = 5)
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, val[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,36] = confusionMatrix(as.factor(rf_pred), val$target)$overall[1]
  kpa.mat[iter,36] = confusionMatrix(as.factor(rf_pred), val$target)$overall[2]
  auc.mat[iter,36] = multiclass.roc(val$target, rf_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target ~ ., data = orsam, ntree = 100, mtry = 6)
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, val[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,37] = confusionMatrix(as.factor(rf_pred), val$target)$overall[1]
  kpa.mat[iter,37] = confusionMatrix(as.factor(rf_pred), val$target)$overall[2]
  auc.mat[iter,37] = multiclass.roc(val$target, rf_pred)$auc[1]
  
  #-----------------------------------------Naive-Bayesian
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #Naive-Bayesian 모델 학습
    nb_model = naiveBayes(orsam[,-ncol(dat)], orsam$target)
    # 예측 수행
    prob_pred[,iiter] = predict(nb_model, val[,-ncol(dat)], type = "raw")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  nb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,38] = confusionMatrix(as.factor(nb_pred), val$target)$overall[1]
  kpa.mat[iter,38] = confusionMatrix(as.factor(nb_pred), val$target)$overall[2]
  auc.mat[iter,38] = multiclass.roc(val$target, nb_pred)$auc[1]
  
  #-----------------------------------------SVM
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #SVM 모델 학습
    svm_model = svm(target~ ., data = orsam, probability = T)
    # 예측 수행
    prob_pred[,iiter] = attr(predict(svm_model, val[,-ncol(dat)], probability = T),"probabilities")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  svm_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,39] = confusionMatrix(as.factor(svm_pred), val$target)$overall[1]
  kpa.mat[iter,39] = confusionMatrix(as.factor(svm_pred), val$target)$overall[2]
  auc.mat[iter,39] = multiclass.roc(val$target, svm_pred)$auc[1]
  
  #-----------------------------------------KNN
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #KNN 모델 학습
    knn_model = knn(orsam[,-ncol(dat)], val[,-ncol(dat)], cl = orsam$target, prob=T)
    # 예측 수행
    knn_prob=attr(knn_model,"prob") ; knn_prob[knn_model==0]=(1-knn_prob[knn_model==0])
    prob_pred[,iiter] = knn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  knn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,40] = confusionMatrix(as.factor(knn_pred), val$target)$overall[1]
  kpa.mat[iter,40] = confusionMatrix(as.factor(knn_pred), val$target)$overall[2]
  auc.mat[iter,40] = multiclass.roc(val$target, knn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #KNN 모델 학습
    knn_model = knn(orsam[,-ncol(dat)], val[,-ncol(dat)], cl = orsam$target, k=3, prob=T)
    # 예측 수행
    knn_prob=attr(knn_model,"prob") ; knn_prob[knn_model==0]=(1-knn_prob[knn_model==0])
    prob_pred[,iiter] = knn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  knn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,41] = confusionMatrix(as.factor(knn_pred), val$target)$overall[1]
  kpa.mat[iter,41] = confusionMatrix(as.factor(knn_pred), val$target)$overall[2]
  auc.mat[iter,41] = multiclass.roc(val$target, knn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #KNN 모델 학습
    knn_model = knn(orsam[,-ncol(dat)], val[,-ncol(dat)], cl = orsam$target, k=5, prob=T)
    # 예측 수행
    knn_prob=attr(knn_model,"prob") ; knn_prob[knn_model==0]=(1-knn_prob[knn_model==0])
    prob_pred[,iiter] = knn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  knn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,42] = confusionMatrix(as.factor(knn_pred), val$target)$overall[1]
  kpa.mat[iter,42] = confusionMatrix(as.factor(knn_pred), val$target)$overall[2]
  auc.mat[iter,42] = multiclass.roc(val$target, knn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #KNN 모델 학습
    knn_model = knn(orsam[,-ncol(dat)], val[,-ncol(dat)], cl = orsam$target, k=10, prob=T)
    # 예측 수행
    knn_prob=attr(knn_model,"prob") ; knn_prob[knn_model==0]=(1-knn_prob[knn_model==0])
    prob_pred[,iiter] = knn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  knn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,43] = confusionMatrix(as.factor(knn_pred), val$target)$overall[1]
  kpa.mat[iter,43] = confusionMatrix(as.factor(knn_pred), val$target)$overall[2]
  auc.mat[iter,43] = multiclass.roc(val$target, knn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #KNN 모델 학습
    knn_model = knn(orsam[,-ncol(dat)], val[,-ncol(dat)], cl = orsam$target, k=15, prob=T)
    # 예측 수행
    knn_prob=attr(knn_model,"prob") ; knn_prob[knn_model==0]=(1-knn_prob[knn_model==0])
    prob_pred[,iiter] = knn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  knn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,44] = confusionMatrix(as.factor(knn_pred), val$target)$overall[1]
  kpa.mat[iter,44] = confusionMatrix(as.factor(knn_pred), val$target)$overall[2]
  auc.mat[iter,44] = multiclass.roc(val$target, knn_pred)$auc[1]
  
  #-----------------------------------------LDA
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #LDA 모델 학습
    lda_model = lda(target ~ ., data = orsam)
    # 예측 수행
    prob_pred[,iiter] = predict(lda_model, val[,-ncol(dat)])$posterior[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lda_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,45] = confusionMatrix(as.factor(lda_pred), val$target)$overall[1]
  kpa.mat[iter,45] = confusionMatrix(as.factor(lda_pred), val$target)$overall[2]
  auc.mat[iter,45] = multiclass.roc(val$target, lda_pred)$auc[1]
  
  #-----------------------------------------Neural Networks
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #Neural Networks 모델 학습
    nn_model = nnet(target ~ ., data = orsam, size = 3, maxit = 100)
    # 예측 수행
    nn_prob = predict(nn_model, val[,-ncol(dat)], type='raw')
    nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0]=(1-nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0])
    prob_pred[,iiter] = nn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  nn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,46] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,46] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,46] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #Neural Networks 모델 학습
    nn_model = nnet(target ~ ., data = orsam, size = 6, maxit = 100)
    # 예측 수행
    nn_prob = predict(nn_model, val[,-ncol(dat)], type='raw')
    nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0]=(1-nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0])
    prob_pred[,iiter] = nn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  nn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,47] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,47] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,47] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #Neural Networks 모델 학습
    nn_model = nnet(target ~ ., data = orsam, size = 9, maxit = 100)
    # 예측 수행
    nn_prob = predict(nn_model, val[,-ncol(dat)], type='raw')
    nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0]=(1-nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0])
    prob_pred[,iiter] = nn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  nn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,48] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,48] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,48] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    #Neural Networks 모델 학습
    nn_model = nnet(target ~ ., data = orsam, size = 12, maxit = 100)
    # 예측 수행
    nn_prob = predict(nn_model, val[,-ncol(dat)], type='raw')
    nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0]=(1-nn_prob[predict(nn_model, val[,-ncol(dat)], type='class')==0])
    prob_pred[,iiter] = nn_prob
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  nn_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,49] = confusionMatrix(as.factor(nn_pred), val$target)$overall[1]
  kpa.mat[iter,49] = confusionMatrix(as.factor(nn_pred), val$target)$overall[2]
  auc.mat[iter,49] = multiclass.roc(val$target, nn_pred)$auc[1]
  
  #-----------------------------------------Adaboost
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # Adaboost 모델 학습
    ab_model = boosting(target ~ ., data = orsam, mfinal = 100)
    # 예측 수행
    prob_pred[,iiter] = predict(ab_model, val[,-ncol(dat)])$prob[,1]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  ab_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,50] = confusionMatrix(as.factor(ab_pred), val$target)$overall[1]
  kpa.mat[iter,50] = confusionMatrix(as.factor(ab_pred), val$target)$overall[2]
  auc.mat[iter,50] = multiclass.roc(val$target, ab_pred)$auc[1]
  
  #-----------------------------------------xgboost
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # XGBoost 모델 학습
    xgb_train = xgb.DMatrix(data = as.matrix(orsam[,-ncol(dat)]), label = orsam$target)
    xgb_params = list(objective = "binary:logistic")
    xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(xgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  xgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,51] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,51] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,51] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # XGBoost 모델 학습
    xgb_train = xgb.DMatrix(data = as.matrix(orsam[,-ncol(dat)]), label = orsam$target)
    xgb_params = list(objective = "binary:logistic", max_depth = 3)
    xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(xgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  xgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,52] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,52] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,52] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # XGBoost 모델 학습
    xgb_train = xgb.DMatrix(data = as.matrix(orsam[,-ncol(dat)]), label = orsam$target)
    xgb_params = list(objective = "binary:logistic", max_depth = 6)
    xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(xgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  xgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,53] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,53] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,53] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # XGBoost 모델 학습
    xgb_train = xgb.DMatrix(data = as.matrix(orsam[,-ncol(dat)]), label = orsam$target)
    xgb_params = list(objective = "binary:logistic", max_depth = 9)
    xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(xgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  xgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,54] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,54] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,54] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # XGBoost 모델 학습
    xgb_train = xgb.DMatrix(data = as.matrix(orsam[,-ncol(dat)]), label = orsam$target)
    xgb_params = list(objective = "binary:logistic", max_depth = 12)
    xgb_model = xgb.train(data = xgb_train, nrounds = 100, params = xgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(xgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  xgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,55] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[1]
  kpa.mat[iter,55] = confusionMatrix(as.factor(xgb_pred), val$target)$overall[2]
  auc.mat[iter,55] = multiclass.roc(val$target, xgb_pred)$auc[1]
  
  #-----------------------------------------lightgbm
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # lightGBM 모델 학습
    lgb_train = lgb.Dataset(data = as.matrix(orsam[, -ncol(dat)]), label = orsam$target)
    lgb_params = list(objective = "binary")
    lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(lgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,56] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,56] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,56] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # lightGBM 모델 학습
    lgb_train = lgb.Dataset(data = as.matrix(orsam[, -ncol(dat)]), label = orsam$target)
    lgb_params = list(objective = "binary", max_depth = 3)
    lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(lgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,57] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,57] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,57] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # lightGBM 모델 학습
    lgb_train = lgb.Dataset(data = as.matrix(orsam[, -ncol(dat)]), label = orsam$target)
    lgb_params = list(objective = "binary", max_depth = 6)
    lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(lgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,58] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,58] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,58] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # lightGBM 모델 학습
    lgb_train = lgb.Dataset(data = as.matrix(orsam[, -ncol(dat)]), label = orsam$target)
    lgb_params = list(objective = "binary", max_depth = 9)
    lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(lgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,59] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,59] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,59] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    # lightGBM 모델 학습
    lgb_train = lgb.Dataset(data = as.matrix(orsam[, -ncol(dat)]), label = orsam$target)
    lgb_params = list(objective = "binary", max_depth = 12)
    lgb_model = lgb.train(data = lgb_train, nrounds = 100, params = lgb_params)
    # 예측 수행
    prob_pred[,iiter] = predict(lgb_model, as.matrix(val[,-ncol(dat)]))
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  lgb_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,60] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[1]
  kpa.mat[iter,60] = confusionMatrix(as.factor(lgb_pred), val$target)$overall[2]
  auc.mat[iter,60] = multiclass.roc(val$target, lgb_pred)$auc[1]
  
  print(iter)
})
# n=1 60개 12분

b.rst.tab[1,] = apply(acc.mat,2,median)
b.rst.tab[2,] = apply(kpa.mat,2,median)
b.rst.tab[3,] = apply(auc.mat,2,median)
b.rst.tab

write.csv(b.rst.tab, 'basic_table.csv')

r=10

#-----------------------------------------randomforest_multi

m_rf_grid = function(dat, r, parm){
  rf.auc.mat = matrix(NA, r, nrow(parm))
  dat$target = as.factor(dat$target)
  set = sample(1:nrow(dat), 0.5*nrow(dat))
  train = dat[set,]
  test = dat[-set,]
  for(iter in 1:r){
    nset = sample(1:nrow(train), 10000)
    osam = sqrt.fun(train[nset,])
    val = sqrt.fun(train[-nset,])
    sam = scaling.minmax(osam,osam)
    val = scaling.minmax(val,osam)
    for(iiter in 1:nrow(parm)){
      g_model = randomForest(target~., data = sam,
                             ntree = parm[iiter,"ntree"], mtry = parm[iiter,"mtry"], nodesize = parm[iiter,"nodesize"])
      g_pred = predict(g_model, val[,-ncol(dat)]) #factor
      rf.auc.mat[iter,iiter] = multiclass.roc(val$target, as.numeric(g_pred))$auc[1]
    }
  }
  plot(apply(rf.auc.mat,2,median), type='b', col='lightseagreen')
  print(cbind(apply(rf.auc.mat,2,median),parm)[order(cbind(apply(rf.auc.mat,2,median),parm)[,1], decreasing = TRUE),])
  opt = parm[which.max(apply(rf.auc.mat,2,median)),]
  return(opt)
}

m_rf_parm = expand.grid(ntree=c(300), mtry = c(3), nodesize = c(1))

system.time({m_rf_opt = m_rf_grid(dat, r, m_rf_parm)})
m_rf_opt
# 1개 20초

#-----------------------------------------xgboost_multi

m_xgb_grid = function(dat, r, parm){
  xgb.auc.mat = matrix(NA, r, nrow(parm))
  set = sample(1:nrow(dat), 0.5*nrow(dat))
  train = dat[set,]
  test = dat[-set,]
  for(iter in 1:r){
    nset = sample(1:nrow(train), 10000)
    osam = sqrt.fun(train[nset,])
    val = sqrt.fun(train[-nset,])
    sam = scaling.minmax(osam,osam)
    val = scaling.minmax(val,osam)
    sam$target = as.numeric(sam$target)-1
    xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
    for(iiter in 1:nrow(parm)){
      params = list(objective = "multi:softmax", num_class = length(unique(dat$target)),
                    eta = parm[iiter,"eta"], max_depth = parm[iiter,"max_depth"])
      g_model = xgb.train(data = xgb_train, nrounds = parm[iiter,"nrounds"], params = params)
      g_pred = predict(g_model, as.matrix(val[,-ncol(dat)])) #numeric
      xgb.auc.mat[iter,iiter] = multiclass.roc(val$target, g_pred)$auc[1]
    }
  }
  plot(apply(xgb.auc.mat,2,median), type='b', col='lightseagreen')
  print(cbind(apply(xgb.auc.mat,2,median),parm)[order(cbind(apply(xgb.auc.mat,2,median),parm)[,1], decreasing = TRUE),])
  opt = parm[which.max(apply(xgb.auc.mat,2,median)),]
  return(opt)
}

m_xgb_parm = expand.grid(eta = c(0.1), max_depth = c(9), nrounds = c(300))

system.time({m_xgb_opt = m_xgb_grid(dat, r, m_xgb_parm)})
m_xgb_opt
# 1개 90초

#-----------------------------------------randomforest_one_vs_rest

or_rf_grid = function(dat, r, parm){
  rf.auc.mat = matrix(NA, r, nrow(parm))
  set = sample(1:nrow(dat), 0.5*nrow(dat))
  train = dat[set,]
  test = dat[-set,]
  prob_pred = matrix(NA, nrow=nrow(train)-10000, ncol=length(unique(dat$target)))
  for(iter in 1:r){
    nset = sample(1:nrow(train), 10000)
    osam = sqrt.fun(train[nset,])
    val = sqrt.fun(train[-nset,])
    sam = scaling.minmax(osam,osam)
    val = scaling.minmax(val,osam)
    orsam = sam
    for(iiter in 1:nrow(parm)){
      for(iiiter in 1:length(unique(dat$target))){
        orsam$target = ifelse(sam$target == (iiiter-1), 1, 0)
        orsam$target = factor(orsam$target, levels=c(1, 0))
        g_model = randomForest(target~., data = orsam,
                               ntree = parm[iiter,"ntree"], mtry = parm[iiter,"mtry"], nodesize = parm[iiter,"nodesize"])
        prob_pred[,iiiter] = predict(g_model, val[,-ncol(dat)], type="prob")[,"1"]
      }
      g_pred = max.col(prob_pred)-1
      rf.auc.mat[iter,iiter] = multiclass.roc(val$target, g_pred)$auc[1]
    }
  }
  plot(apply(rf.auc.mat,2,median), type='b', col='lightseagreen')
  print(cbind(apply(rf.auc.mat,2,median),parm)[order(cbind(apply(rf.auc.mat,2,median),parm)[,1], decreasing = TRUE),])
  opt = parm[which.max(apply(rf.auc.mat,2,median)),]
  return(opt)
}

or_rf_parm = expand.grid(ntree=c(200), mtry = c(2), nodesize = c(1))

system.time({or_rf_opt = or_rf_grid(dat, r, or_rf_parm)})
or_rf_opt
# 1개 105초

f.rst.tab = matrix(NA, 3, 3)
rownames(f.rst.tab) = c("acc", "kpa", "auc")
colnames(f.rst.tab) = c('m_rf', 'or_rf', 'or_lgb')

n=30
acc.mat = kpa.mat = auc.mat = matrix(NA,n,3)

system.time(for(iter in 1:n){
  dat$target = as.factor(dat$target)
  set = sample(1:nrow(dat), 0.5*nrow(dat))
  train = dat[set,]
  test = sqrt.fun(dat[-set,])
  
  nset = sample(1:nrow(train), 10000)
  osam = sqrt.fun(train[nset,])
  sam = scaling.minmax(osam,osam)
  test = scaling.minmax(test,osam)
  
  #-----------------------------------------randomforest_multi
  # randomforest 모델 학습
  rf_model = randomForest(target~., data = sam,
                          ntree = m_rf_opt[1,"ntree"], mtry = m_rf_opt[1,"mtry"], nodesize = m_rf_opt[1,"nodesize"])
  # 예측 수행
  rf_pred = predict(rf_model, test[,-ncol(dat)]) #factor
  # 예측 결과 평가
  acc.mat[iter,1] = confusionMatrix(rf_pred, test$target)$overall[1]
  kpa.mat[iter,1] = confusionMatrix(rf_pred, test$target)$overall[2]
  auc.mat[iter,1] = multiclass.roc(test$target, as.numeric(rf_pred)-1)$auc[1]
  
  #------------------------------------------타겟을 numeric
  sam$target = as.numeric(sam$target)-1
  
  #-----------------------------------------xgboost_multi
  xgb_train = xgb.DMatrix(data = as.matrix(sam[,-ncol(dat)]), label = sam$target)
  # XGBoost 모델 학습
  xgb_params = list(objective = "multi:softmax", num_class = length(unique(dat$target)),
                    eta = m_xgb_opt[1,"eta"], max_depth = m_xgb_opt[1,"max_depth"])
  xgb_model = xgb.train(data = xgb_train, nrounds = m_xgb_opt[1,"nrounds"], params = xgb_params)
  # 예측 수행
  xgb_pred = predict(xgb_model, newdata = as.matrix(test[,-ncol(dat)])) #numeric
  # 예측 결과 평가
  acc.mat[iter,2] = confusionMatrix(as.factor(xgb_pred), test$target)$overall[1]
  kpa.mat[iter,2] = confusionMatrix(as.factor(xgb_pred), test$target)$overall[2]
  auc.mat[iter,2] = multiclass.roc(test$target, xgb_pred)$auc[1]
  
  #-----------------------------------------one_vs_rest
  osam = sqrt.fun(train[nset,])
  test = sqrt.fun(dat[-set,])
  sam = scaling.minmax(osam,osam)
  test = scaling.minmax(test,osam)
  
  orsam = sam
  
  prob_pred = matrix(NA, nrow=nrow(test), ncol=length(unique(dat$target)))
  #-----------------------------------------randomforest_one_vs_rest
  for(iiter in 1:length(unique(dat$target))){
    orsam$target = ifelse(sam$target == (iiter-1), 1, 0)
    orsam$target = factor(orsam$target, levels=c(1, 0))
    # randomforest 모델 학습
    rf_model = randomForest(target~., data = orsam,
                            ntree = or_rf_opt[1,"ntree"], mtry = or_rf_opt[1,"mtry"], nodesize = or_rf_opt[1,"nodesize"])
    # 예측 수행
    prob_pred[,iiter] = predict(rf_model, test[,-ncol(dat)], type="prob")[,"1"]
  }
  # 가장 높은 확률 값을 가진 클래스 선택
  rf_pred = max.col(prob_pred)-1
  # 예측 결과 평가
  acc.mat[iter,3] = confusionMatrix(as.factor(rf_pred), test$target)$overall[1]
  kpa.mat[iter,3] = confusionMatrix(as.factor(rf_pred), test$target)$overall[2]
  auc.mat[iter,3] = multiclass.roc(test$target, rf_pred)$auc[1]
  
  print(iter)
})
f.rst.tab[1,] = apply(acc.mat,2,median)
f.rst.tab[2,] = apply(kpa.mat,2,median)
f.rst.tab[3,] = apply(auc.mat,2,median)
f.rst.tab

write.csv(f.rst.tab, 'final_table.csv')
