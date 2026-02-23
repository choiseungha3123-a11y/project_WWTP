# application.properties : 공용 설정
# 어플리케이션 명칭
spring.application.name=aiprojectserver
# 사용 포트
server.port=8081

# 어플리케이션 내부 시간기준
spring.timezone=Asia/Seoul
# API를 통한 JSON 구성시의 시간기준
spring.jackson.time-zone=Asia/Seoul
# DB에서의 시간기준
spring.jpa.properties.hibernate.jdbc.time_zone=Asia/Seoul

#Mysql 설정
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.jpa.database-platform=org.hibernate.dialect.MySQL8Dialect
spring.jpa.database=mysql

# 비공개 데이터 로드
spring.profiles.include=db, oauth2, api

# JPA Setting
spring.jpa.hibernate.ddl-auto=update
#spring.jpa.show-sql=true
#spring.jpa.properties.hibernate.format_sql=true

# APIHUB로부터 날씨정보 수집 여부
scheduler.enable=false
# 예측에 사용할 임의 날짜 처리 시간
# at 0:05
scheduler.fakeday.cron=0 27,57 * * * *
# APIHUB로부터 날씨정보 수집 시간
# at 28 and 58 minutes
scheduler.gether.cron=0 28,58 * * * *
# 수집된 날씨정보의 완결성(하루에 분당 1회 1440개) 체크 시간
# at 29 and 59 minutes
scheduler.complete.cron=0 29,59 * * * *
# 예측조회 여부
predict.enable=false
# 예측조회 시간
#predict every 30 minute
scheduler.predict.cron=0 0,30 * * * *
# 보고서 작성 여부
report.enable=false
# 보고서 작성 시간
#every 6AM, 6PM send report mail
scheduler.report.cron=50 0 6,18 * * *

#CORS Allowed Origin Patterns
#spring.AllowedOriginPatterns=http://localhost:80,http://127.0.0.1:80, https://www.projectwwtp.kro.kr, http://localhost:3000, http://127.0.0.1:3000, https://www.projectwwtp.kro.kr:3000
spring.AllowedOriginPatterns=*

#File Upload Size
spring.servlet.multipart.max-file-size=30MB
spring.servlet.multipart.max-request-size=30MB

#OAuth2
spring.security.oauth2.client.registration.naver.client-name=Naver
spring.security.oauth2.client.registration.naver.authorization-grant-type=authorization_code
spring.security.oauth2.client.provider.naver.authorization-uri=https://nid.naver.com/oauth2.0/authorize
spring.security.oauth2.client.provider.naver.token-uri=https://nid.naver.com/oauth2.0/token
spring.security.oauth2.client.provider.naver.user-info-uri=https://openapi.naver.com/v1/nid/me
spring.security.oauth2.client.provider.naver.user-name-attribute=response

spring.security.oauth2.client.registration.kakao.client-authentication-method=client_secret_post
spring.security.oauth2.client.registration.kakao.client-name=Kakao
spring.security.oauth2.client.registration.kakao.authorization-grant-type=authorization_code
spring.security.oauth2.client.provider.kakao.authorization-uri=https://kauth.kakao.com/oauth/authorize
spring.security.oauth2.client.provider.kakao.token-uri=https://kauth.kakao.com/oauth/token
spring.security.oauth2.client.provider.kakao.user-info-uri:https://kapi.kakao.com/v2/user/me
spring.security.oauth2.client.provider.kakao.user-name-attribute:id


# application-db.properties : DB 관련 설정
# DB접속설정
spring.datasource.url=
# DB접속 ID
spring.datasource.username=
# DB접속 비밀번호
spring.datasource.password=
# 개인정보 암호화키
db.cryp.key=
# 개인정보 암호화 초기화 백터
# 추후 무작위생성하고 암호화 함께 처리하도록 수정 필요
db.cryp.iv=

# application-oauth2.properties : 소셜 로그인 관련 설정
# 소셜로그인 후 보여질 페이지 주소
spring.auth2.URI=
# 소셜로그인을 위한 클라이언트 ID
spring.security.oauth2.client.registration.google.client-id=
spring.security.oauth2.client.registration.naver.client-id=
spring.security.oauth2.client.registration.kakao.client-id=
# 소셜로그인을 위한 비밀키
spring.security.oauth2.client.registration.google.client-secret=
spring.security.oauth2.client.registration.naver.client-secret=
spring.security.oauth2.client.registration.kakao.client-secret=
# 소셜로그인을 위한 정보 조회범위
spring.security.oauth2.client.registration.google.scope=
spring.security.oauth2.client.registration.naver.scope=
spring.security.oauth2.client.registration.kakao.scope=
# 소셜로그인 토큰을 전달받을 페이지
spring.security.oauth2.client.registration.google.redirect-uri=
spring.security.oauth2.client.registration.naver.redirect-uri=
spring.security.oauth2.client.registration.kakao.redirect-uri=
