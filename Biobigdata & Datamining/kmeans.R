rm(list = ls())

data(wine, package="rattle.data")
head(wine)
=
#������ ����ȭ�� �������� ������ ǥ��ȭ�ϴ� ���� �ǹ� 
#ǥ��ȭ�� �������� �������� ����� ���ų� 
#������ ��ü �������� ǥ�� ������ ������ �۾��� ���� 
#�������� ����� 0�� �ǰ� ���� ���� ����(����) ���� ����������.
df <- scale(wine[-1])  #without the first column

#���� ���� ���� ����-�� ������(within-groups sum of squares)�� �׷���
#data�� ��ġ���� �ڷ��̸�, nc�� ������ ������ �ִ� ��, seed�� �����߻� �ʱⰪ
wssplot <- function(data, nc=15, seed=1234){ 
  wss <- (nrow(data)-1)*sum(apply(data,2,var)) 
   for (i in 2:nc){
                set.seed(seed)
                wss[i] <- sum(kmeans(data, centers=i)$withinss)
  } 
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
}

#���� ���� ���ϱ� ����
wssplot(df)
#���� �� 3���� ������������ ũ�� ���ҵǾ����� Ȯ��

#������ ��(k)�� 3���� �Ͽ� kmeans()�� ������ ���
#�� ������ ũ��� �߽ɰ��� �����ش�. 
#���� ����� �ð�ȭ�� plot() �Լ��� �̿�
#nstart= generate 25 initial random centroids 

set.seed(1234)
fit.km <- kmeans(df, 3, nstart=25)
fit.km$size 

fit.km$centers

# ���������  �ð�ȭ
plot(df, col=fit.km$cluster)

#�� ������ �߽����߰�
points(fit.km$center, col=1:2, pch=8, cex=1.5)

# ���������  ���: (ǥ��ȭ��  �ڷῡ  ����)��������� ���ڷ��� ������ ��ȯ
aggregate(wine[-1], by=list(cluster=fit.km$cluster), mean) 

# ���������  ����: �����з�ǥ
ct.km <- table(wine$Type, fit.km$cluster)
ct.km