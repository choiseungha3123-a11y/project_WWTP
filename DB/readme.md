DB URL : miniproject.c70cou4sk2dq.ap-northeast-2.rds.amazonaws.com:3306
DB Schema : aiproject
User : aiproject
Password : projectwwtp


-실행중인 JAR 확인-
ps -ef | grep jar
kill XXXX(숫자, 프로세스 ID)
-jar 파일(서버)를 back에서 돌아가도록하는 명령(메모리를 1G로 할당)-
nohup java -Xmx1G -jar aiprojectserver-0.0.1.jar > ai-server.log 2>&1 &
-MYSQL 관련-
mysql -u root -p
sudo systemctl status mysqld
-nextjs 빌드-
npm run build
-pm2 실행 관련-
pm2 start npm --name "miniproject-app" -- start
pm2 restart all
pm2 list
-NGINX(포트 포워딩용) 관련-
sudo systemctl enable nginx
sudo systemctl start nginx
sudo vi /etc/nginx/nginx.conf
sudo systemctl restart nginx
-파이썬 관련-
active 폴더 찾기(/opt/cert/bin/activate)
sudo find / -name "activate" -type f | grep "/bin/"

-배포-

http://10.125.121.176:8081/swagger-ui/index.html




해야할 일 : 
	HTTP 인증 및 도메인 처리
	기상 데이터(강수량, 기온, 습도)를 DB로 저장
	 - 24년부터 가져오되... 1일 단위로 
	 - 현재 기준이 되면 30분(혹은 1시간) 단위로 변경
	서버상에 파이썬 세팅
	amazone linux 2023 let's encrypt
	vscode 안티그라비티

	Oauth2 기능 추가 (구글, 네이버, 카카오)
	메모 처리(한꺼번에 보내기)
	로그인 실패 관련 이력 처리
	조회 로그
	이상 조회 탐지(별도 테이블로 관리)
	날씨 데이터 조회 기록 추가
	날씨 데이터 직접 수정, 삭제(변경 이력 관리)
	데이터 품질(결측/이상치)


완료한 일 :
	강의실에서 리눅스 서버 연결
	강의실에서 DB 서버 연결
	집에서 리눅스 서버 연결
	집에서 DB 서버 연결
	기상청 API 허브 가입 및 인증키 발급
	swagger UI를 통한 API 설명 페이지 구성
	로그인 토큰 처리 추가
	회원 관리(로그인/추가/변경/삭제)
	회원 정보에 이름 추가
	ID 중복 확인 / (X)비밀번호 제한 추가(비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.)
	동시 로그인 제한
	DB 이전
	