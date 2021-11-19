rm(list = ls())

library(adabag)
data(iris)
boo.adabag <- boosting(Species~., data=iris, boos=TRUE, mfinal=10)
boo.adabag$importance

#{adabag}�� boosting() �Լ��� �ν����� �̿��Ͽ� �з��� ����
#plot() �Լ��� ���� �з� ����� Ʈ�� ���·� ���

plot(boo.adabag$trees[[10]])
text(boo.adabag$trees[[10]])


#predict() �Լ��� ���� ���ο� �ڷῡ ���� �з�(����)�� ����

pred <- predict(boo.adabag, newdata=iris)
tb <- table(pred$class, iris[,5])
tb
#setosa, versicolor, virginica ��� ��Ȯ�� �з�


#���з��� ���
error.rpart <- 1-(sum(diag(tb))/sum(tb))
error.rpart 