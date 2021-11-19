rm(list = ls())

library(adabag)
data(iris)

iris.bagging <- bagging(Species~., data=iris, mfinal=10)
# mfinal= �ݺ���  �Ǵ�  Ʈ����  ��(����Ʈ=100)

iris.bagging$importance	# ������  �������  �߿䵵

#������  �߿䵵��  ��  Ʈ������  ������  ����  �־�����  
#����������  gain(�Ǵ� ��Ȯ�Ǽ���  ���ҷ�)��  ������  ô��


#R ��Ű�� {adabag}�� bagging() �Լ��� ����� �̿��Ͽ� �з��� ����
#plot() �Լ��� ���� �з� ����� Ʈ�� ���·� ���

plot(iris.bagging$trees[[10]])
text(iris.bagging$trees[[10]])

#predict() �Լ��� ���� ���ο� �ڷῡ ���� �з�(����)�� ����
#���� ���࿡ ���� �ڷḦ �����Ͽ� �з��� ����
 
pred <- predict(iris.bagging, newdata=iris)
table(pred$class, iris[,5])
 

