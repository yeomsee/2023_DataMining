# 필요한 패키지
install.packages("lubridate") ; library(lubridate)
install.packages('dplyr') ; library(dplyr)

# kt 데이터
delivery1 = read.csv("KGU_3rd_ORIGIN_KGUWTHRDLVRDF_20190701000000.csv", header = F)
delivery2 = read.csv("KGU_3rd_ORIGIN_KGUWTHRDLVRDF_20200101000000.csv", header = F)
delivery = rbind(delivery1,delivery2)
names(delivery) = c('시도', '시군구', '날짜', '시간대', '강수여부', '습도', '강수량', '기온',
                    '풍속', '바람강도', '바람유형', '풍향값', '풍향카테고리', '한식',
                    '분식', '카페/디저트', '돈까스/일식', '회', '치킨', '피자', '아시안/양식',
                    '중식', '족발/보쌈', '야식', '찜탕', '도시락', '패스트푸드')
# 미세먼지 데이터
air1 = read.csv('기간별_일평균_대기환경_정보_2019년.csv', header = T, fileEncoding = "euc-kr")
air2 = read.csv('기간별_일평균_대기환경_정보_2020년.csv', header = T, fileEncoding = "euc-kr")
air = rbind(air1,air2)
names(air) = c('날짜', '권역명', '시군구', '이산화질소농도', '오존농도', '일산화탄소농도',
               '아황산가스농도', '미세먼지농도', '초미세먼지농도')

# Kt + 미세먼지 데이터 통합
delivery$날짜 = ymd(delivery$날짜)
air$날짜 = ymd(air$날짜)
df = left_join(delivery, air, by = c('날짜', '시군구'))
df = subset(df, 시도=='서울특별시')

rownames(df) = NULL

# 종속변수 정리
data = df
data$한식 = df$한식 + df$찜탕
data$일식 = df$돈까스.일식 + df$회

# 필요없는 컬럼 삭제
data = subset(data, select = -c(시도, 시군구, 날짜, 시간대, 권역명))
data = subset(data, select = -c(돈까스.일식, 회, 찜탕, 야식, 도시락))
data = subset(data, select = -c(바람유형, 바람강도, 풍향카테고리))
str(data)

summary(data)

# 미세먼지 관련 데이터 NA 제거
data=na.omit(data)

# -1 값이 존재하는 데이터 제거
data = subset(data, data$습도 & data$풍속 & data$풍향값 >= 0)
data = subset(data, data$습도 >=0)

# 독립변수, 종속변수 
data = data %>% relocate(c("한식","분식","카페.디저트","치킨","피자","아시안.양식","중식","족발.보쌈","패스트푸드"), .before = "일식")

# 독립변수 수치형으로 변환
data$강수여부 = as.numeric(as.factor(data$강수여부))-1 # 눈=0, 비=1, 없음=2, 진눈개비=3

select.y=c("한식","분식","카페.디저트","치킨","피자","아시안.양식","중식","족발.보쌈","패스트푸드","일식")

target= as.character(rep(1,nrow(data)))
for (id in 1:nrow(data)){target[id]=names(which.max(data[id,select.y]))}

target =as.data.frame(target)
data = subset(data, select = -c(한식,분식,카페.디저트,치킨,피자,아시안.양식,중식,족발.보쌈,패스트푸드,일식))

data= data[,-c(한식,분식,카페.디저트,치킨,피자,아시안.양식,중식,족발.보쌈,패스트푸드,일식)]

data = cbind(data, target)

data$target = as.numeric(as.factor(data$target))-1
#분식=0, 아시안.양식=1, 일식=2,  족발.보쌈=3, 중식=4, 치킨=5, 카페.디저트=6, 패스트푸드=7, 피자=8,  한식=9 

write.csv(data, "data_max.csv",  row.names = FALSE, fileEncoding = "euc-kr")