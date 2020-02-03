library('pROC')

y_true = read.csv('./Research/PRUV/true_nih.csv', header=FALSE)$V6
y_pred = read.csv('./Research/PRUV/pred_nih.csv', header=FALSE)$V6

roc = roc(y_true, y_pred)
roc

ci.auc(roc)


y_true = read.csv('./Research/PRUV/true_stf.csv', header=FALSE)$V6
y_pred = read.csv('./Research/PRUV/pred_stf.csv', header=FALSE)$V6

roc = roc(y_true, y_pred)
roc

ci.auc(roc)
