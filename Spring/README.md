🚀 Amazon Linux 2023 서버 설정 가이드

Amazon Linux 2023(AL2023) 인스턴스 생성후 인바운드 규칙(80:http, 443:https, 22:ssl, 3000:nextjs, 8080:springboot, 8000:fastpai 등) 추가

1\. 가상메모리 (Swap File) 설정

RAM 부족으로 인한 프로세스 다운을 방지하기 위해 2GB 스왑 공간을 할당합니다.

\#가상 메모리 할당

sudo dd if=/dev/zero of=/swapfile bs=128M count=16

\#권한 설정

sudo chmod 600 /swapfile

\# 스왑 영역 설정

sudo mkswap /swapfile

\# 스왑 영역 활성화

sudo swapon /swapfile

\# 재부팅시 자동 활성화

echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

\# 설정 확인

sudo swapon -s



2\. Nginx 설정

AL2023의 기본 패키지 관리자인 dnf를 사용합니다.

\#설치

sudo dnf install nginx -y

\#실행

sudo systemctl enable --now nginx

\#설정 변경

sudo vi /etc/nginx/nginx.conf

\#설정은 HTTPS를 위한 443만을 설정

\#기본 연결은 NextJS의 3000포트로 연결하고

\#/api/는 8080포트로 연결 되도록 설정

\#변경 내용 검증

sudo nginx -t

\#변경후 재시작

sudo systemctl restart ngix



3\. MySQL 설정

\#다운 및 설치

sudo dnf install https://dev.mysql.com/get/mysql80-community-release-el9-5.noarch.rpm -y

\# (윈도우에서 다운 후 sftp로 파일 업로드)설치

sudo dnf install mysql-community-server -y

\# 실행

sudo systemctl enable --now mysqld

\#임시 비밀번호 확인

sudo grep 'temporary password' /var/log/mysqld.log

\#보안 설정

sudo mysql\_secure\_installation

\#설치 확인(버전확인)

MySQL -V



4\. Java (OpenJDK) 설정

\#설치 (Java 21 기준)

sudo dnf install java-21-amazon-corretto-devel -y

\#설치 확인(버전 확인)

java -version



5\. Python 설정

\#파이썬 3.14 설치
sudo dnf install -y python3.14 python3.14-pip

\#파이썬 버전 확인

python3.14 --version

\#가상환경 miniconda 설치 (윈도우에서 다운 후 sftp로 파일 업로드)설치

bash Miniconda3-latest-Linux-aarch64.sh

\#miniconda 버전 확인

conda --version

\#가상환경 설정

conda create -n {명칭} python=3.14

\#활성화(활성화 되어야 uvicorn이 실행됨

conda activate {명칭}

\#필요 라이브러리 설치 (torch의 경우 용량이 매우 커서 빼고 설치)

pip install numpy pandas seaborn scikit-learn torch fastapi uvicorn

\# FastAPI 실행

uvicorn main:app --host 0.0.0.0 --port 8000 --reload



6\. Next.js

\#Node.js 설치

sudo dnf install nodejs -y

\#소스를 sftp를 통해 업로드후

\#라이브러리 설치

npm install

\# 개발용 실행

npm run dev

\# 배포용 빌드

npm run build

\# 배포

npm start



7.PM2

\# 프로세스 관리를 위한 프로그램(백그라운드에서 실행되어 관리가 용이)

\#PM2 설치

sudo npm install -g pm2

\# 등록한 프로세스 리스트 확인

pm2 list

\# 프로세스 중지

pm2 stop (name or id)

\# 프로세스 제거

pm2 delete (name or id)

\# 프로세스 재시작

pm2 restart (name or id) --update-env

\# NextJS 배포

pm2 start npm --name "FlowWater-app" -- start

pm2 start npm --name "FlowWater-app-dev" -- run dev

\# Spring 배포 (1G 메모리 옵션 추가 -Xmx1G)

pm2 start "java -Xmx1G -jar aiprojectserver-0.0.1.jar" --name "FlowWater-server" --output "./FlowWater-Server-out.log"

\# FastApi 배포

pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name "FlowWater-Fastapi" --output "./FlowWater-Fastapi-out.log" --error "./FlowWater-Fastapi-error.log"



8\. 도메인 및 SSL 인증서 (Certbot)

\#무료 도메인 등록

https://내도메인.한국/

\#AWS Route 53 연결

\#Let's Encrypt에서 인증서를 발급받는 경우(실패)

https://letsencrypt.org/ko/

\#Certbot 설치:

sudo dnf install python3-certbot-nginx -y

\#인증서 발급:

\# 참조 블로그
https://jun-codinghistory.tistory.com/651
sudo certbot certonly -d \*.도메인네임.???.??? --manual --preferred-challenges dns

# 발급 성공시 인증서의 자동 저장 위치

/etc/letsencrypt/live/projectwwtp.kro.kr/fullchain.pem
/etc/letsencrypt/live/projectwwtp.kro.kr/privkey.pem

\#자동 갱신:
sudo certbot renew --dry-run

\#ZeroSSL에서 인증서를 발급 받는 경우

https://zerossl.com/

\#발급받은 인증서를 다운로드후 압축 해제

\#ca\_bundle.crt, certificate.crt, private.key 파일 확인후 sftp를 통해 업로드

\#crt 파일 병합

cat certificate.crt ca\_bundle.crt > nginx\_ssl.crt

\# 파일 이동
sudo mv nginx\_ssl.crt /etc/pki/nginx/
sudo mv privatekey /etc/pki/nginx/private/

\#설정 변경

sudo vi /etc/nginx/nginx.conf

\#변경 내용 검증

sudo nginx -t

\#변경후 재시작

sudo systemctl restart nginx



9.메일서버 구축
# 시스템 업데이트

sudo dnf update -y

\# Sendmail 및 필수 패키지 설치

sudo dnf install -y sendmail sendmail-cf m4

\# Sendmail 서비스 활성화

sudo systemctl enable sendmail



\# 127.0.0.1 부분을 찾아서 주석 처리하거나 변경

dnl DAEMON\_OPTIONS(`Port=smtp,Addr=127.0.0.1, Name=MTA')dnl

\# 위 줄을 아래처럼 변경 (외부 접속 허용)

DAEMON\_OPTIONS(`Port=smtp, Name=MTA')dnl

\# 도메인 설정 추가

MASQUERADE\_AS(`projectwwtp.kro.kr')dnl

FEATURE(masquerade\_envelope)dnl

FEATURE(masquerade\_entire\_domain)dnl



\# sendmail.cf 재생성

sudo sh -c "m4 /etc/mail/sendmail.mc > /etc/mail/sendmail.cf"



\# local-host-names 설정

sudo vi /etc/mail/local-host-names

\*\*추가할 내용:\*\*

```

projectwwtp.kro.kr

www.projectwwtp.kro.kr

mail.projectwwtp.kro.kr

localhost



\# admin 사용자 생성 (시스템 계정)

sudo useradd -m -s /sbin/nologin admin



\# 비밀번호 설정

sudo passwd admin



\# 메일 디렉토리 권한 확인

sudo mkdir -p /var/spool/mail

sudo chmod 1777 /var/spool/mail



\# firewalld 설치

sudo dnf install -y firewalld



\# 서비스 시작 및 활성화

sudo systemctl start firewalld

sudo systemctl enable firewalld



\# 포트 25 (SMTP) 오픈

sudo firewall-cmd --permanent --add-service=smtp

sudo firewall-cmd --reload



\# Sendmail 시작

sudo systemctl start sendmail

sudo systemctl status sendmail



\# Sendmail 시작

sudo systemctl start sendmail

sudo systemctl status sendmail



\# 메일 큐 확인

sudo mailq# mailx 패키지 설치

sudo dnf install -y mailx





\# 테스트 메일 발송

echo "테스트 메일입니다" | mail -s "테스트" admin@projectwwtp.kro.kr



\# 메일 최근로그 확인

sudo journalctl -u sendmail -n 50

\# 메일 전체 로그 확인

sudo journalctl | grep -i mail



\# 메일함 확인

sudo mail -u admin



