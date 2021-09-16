findwords <- function(tf) {
  
  #txt<-scan(tf,"")       #������ �а�, ������ ������ �����Ͽ� ���͸� ����
  wl<-list()               #�Լ����� ��ȯ�� ����Ʈ ����
  for(i in 1:length(tf)){
    wrd<-tolower(tf[i])   #tf�� ����� �ܾ �ҹ���ȭ
    wl[[wrd]]<-c(wl[[wrd]], i)  #����Ʈ �ε��� ����Ͽ� wl�� �ܾ�� ��ġ�� ����
    print(wrd)
    print(wl[[wrd]])
  }
  return(wl)
  
}

 
freqwl<-function(wrdlist){ # �ܾ� ����Ʈ�� ���Ե� �ܾ��� ������ ���Ϳ� ����
  freqs<-sapply(wrdlist, length)  # length �Լ��� ����Ʈ�� �����Ͽ�, �� �ܾ��� �󵵼��� Ȯ��
  freqs
  return(wrdlist[order(freqs)])  #order(): ������ ���ĵ� ���� ���� �ε����� 
                                 #�󵵼��� ������������ ���� �� ��ȯ
}

#fw<-findwords("C:/Temp/R/findwords_data.txt")
#txt<-scan(tf,"")  in findwords

#�ܺ� ������ ���ε�
data = scan("C:/Users/kate1/Desktop/git-repo/EWHA-Projects/Biobigdata & Datamining/hw-1/obama_speech.txt","")
#Ư������ �����ϱ�
pre.data = gsub("[[:punct:]]", "", data)
str(pre.data)
#fw���� �ܾ�� ��ġ�� �����
fw<-findwords(pre.data)
freq<-freqwl(fw)  # freq �� �󵵼� ������ ���ĵ� ����Ʈ
freq
sfreq<-sapply(freq, length)  #freq���� �� �ܾ��� �󵵼�(����) ǥ��

nwords<-length(sfreq)  # unique �ܾ��� ����
barplot(sfreq[round(0.9*nwords):nwords]) #������ ��Ÿ�� �ܾ��� �󵵼��� ���� 10%�� plotting