rm(list = ls())
data(USArrests)
str(USArrests)

# ������  ����  ����
d <- dist(USArrests, method="euclidean")  # �ֵ�  ����  ��Ŭ����  �Ÿ�
fit <- hclust(d, method="ave")

#dist()�� �Ÿ�(�Ǵ� �����缺)����� �����ϴ� �Լ�
#method= �ɼ��� ���� �پ��� ������� �Ÿ��� ����
#method= �ɼǿ��� "euclidean", "manhattan", "minkowski" ��

#hclust() �Լ��� ������ �����м��� �����ϴ� �Լ�
#method= �ɼ��� ���� ����(�Ǵ� ����) ����� ����
#method= �ɼǿ��� "ward", "single", "complete", "average", "centroid" ��



# ���������  �ð�ȭ: plot() �Լ��� ���� ����α׷����� �ð�ȭ 
plot(fit)


rect.hclust(fit, k=6, border="red")
#plot() �Լ��� �̿��� �׸� ����α׷��� rect.hclust() �Լ��� �̿��Ͽ� 
#�׷��� �簢������ �������� ǥ��. 


rect.hclust(fit, h = 50, which = c(2,3), border = 5:6)
# ����(h) 50����  cut ����, 2��  3����  �簢��  �߰�, �׵θ�  ����(border) ����

#rect.hclust() �Լ��� �׷��(k)�� �̿��Ͽ� �׷��� �ð�ȭ �� �� �ƴ϶�,
# tree�� ����(h)�� ��ġ(which)�� �̿��Ͽ� �׷��� ��ü �Ǵ� �Ϻθ� ǥ��.

###############################################################################

library(cluster)
# stand=TRUE��  �����缺(�Ÿ�)��  ���  ����  ǥ��ȭ��  ������. 
#�Է� �ڷᰡ  �����缺  �����  ��쿡��  ���õ�
(agn1 <- agnes(USArrests, metric="manhattan", stand=TRUE))
plot(agn1)

# diss=TRUE��  �Է�  �ڷᰡ  �����缺  ���  �Ǵ�  dist ��ü��  ��쿡��  ����Ʈ��
agn2 <- agnes(daisy(USArrests), diss=TRUE, method="complete")
plot(agn2)
