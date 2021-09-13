findwords <- function(tf) {
  
  #txt<-scan(tf,"")       #파일을 읽고, 구분자 단위로 구분하여 벡터를 생성
  wl<-list()               #함수에서 반환할 리스트 생성
  for(i in 1:length(tf)){
    wrd<-tolower(tf[i])   #tf에 저장된 단어를 소문자화
    wl[[wrd]]<-c(wl[[wrd]], i)  #리스트 인덱싱 사용하여 wl에 단어와 위치를 저장
    print(wrd)
    print(wl[[wrd]])
  }
  return(wl)
  
}

 
freqwl<-function(wrdlist){ # 단어 리스트에 포함된 단어의 개수를 벡터에 저장
  freqs<-sapply(wrdlist, length)  # length 함수를 리스트에 적용하여, 각 단어의 빈도수를 확인
  freqs
  return(wrdlist[order(freqs)])  #order(): 벡터의 정렬된 값에 대한 인덱스를 
                                 #빈도수를 오름차순으로 정렬 후 반환
}

#fw<-findwords("C:/Temp/R/findwords_data.txt")
#txt<-scan(tf,"")  in findwords

#외부 데이터 업로드
data = scan("C:/Users/kate1/Desktop/git-repo/EWHA-Projects/Biobigdata & Datamining/hw-1/obama_speech.txt","")
#특수문자 제거하기
pre.data = gsub("[[:punct:]]", "", data)
str(pre.data)
#fw에는 단어와 위치가 저장됨
fw<-findwords(pre.data)
freq<-freqwl(fw)  # freq 는 빈도수 순으로 정렬된 리스트
freq
sfreq<-sapply(freq, length)  #freq에서 각 단어의 빈도수(길이) 표시

nwords<-length(sfreq)  # unique 단어의 갯수
barplot(sfreq[round(0.9*nwords):nwords]) #문서에 나타난 단어의 빈도수가 높은 10%를 plotting
