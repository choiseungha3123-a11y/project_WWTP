-실행중인 JAR 확인-
ps -ef | grep jar
kill XXXX(숫자, 프로세스 ID)
-jar 파일(서버)를 back에서 돌아가도록하는 명령(메모리를 1G로 할당)-
pm2 start "java -jar backend-0.0.1-SNAPSHOT.jar" --name "miniproject-server" --output "./miniproject-Server-out.log"
pm2 start "java -Xmx1G -jar aiprojectserver-0.0.1.jar" --name "FlowWater-server" --output "./FlowWater-Server-out.log"

-MYSQL 관련-
mysql -u root -p
sudo systemctl restart mysqld
sudo systemctl status mysqld

-nextjs 빌드-
npm install
npm run build
-pm2 실행 관련-
pm2 start npm --name "miniproject-app" -- start -- -p 3001
pm2 start npm --name "FlowWater-app" -- start
pm2 start npm --name "FlowWater-app-dev" -- run dev
pm2 restart 6 --update-env
pm2 restart all
pm2 list

-NGINX(포트 포워딩용) 관련-
sudo systemctl enable nginx
sudo systemctl start nginx
sudo vi /etc/nginx/nginx.conf
sudo systemctl restart nginx

-파이썬 관련-
conda activate projectwwtp 활성화 후
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name "FlowWater-Fastapi" --output "./FlowWater-Fastapi-out.log" --error "./FlowWater-Fastapi-error.log"

-HTTPS 인증 관련-
sudo certbot certonly -d *.projectwwtp.kro.kr --manual --preferred-challenges dns

-사용중인 포트 확인-
sudo ss -tunlp

http://10.125.121.176:8081/swagger-ui/index.html





해야할 일 : 
	현재 /api/tmsOriginal/imputate 에서 날짜를 지정해서 데이터를 보정 후 FastAPI로 호출할수 있도록 구성했고

	TMS 관련 로그 추가

	이상 조회 탐지(별도 테이블로 관리)

	2024/4/2일까지 확인했음



완료한 일 :
-2026-01-22	아마존 리눅스 설정 시작
-2026-01-23	강의실에서 리눅스 서버 연결
-2026-01-23	강의실에서 DB 서버 연결
-2026-01-23	집에서 리눅스 서버 연결
-2026-01-23	집에서 DB 서버 연결
-2026-01-23	기상청 API 허브 가입 및 인증키 발급
-2026-01-23	swagger UI를 통한 API 설명 페이지 구성
-2026-01-23	로그인 토큰 처리 추가
-2026-01-26	회원 관리(로그인/추가/변경/삭제)
-2026-01-26	회원 정보에 이름 추가
-2026-01-26	ID 중복 확인 / (X)비밀번호 제한 추가(비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.)
-2026-01-26	동시 로그인 제한
-2026-01-27	DB 이전
-2026-01-27	BackEnd, FrontEnd, Python FastAPI가 각각 리눅스 서버에서 연동되는지 확인
-2026-01-28	회원정보 변경시(ID 중복에 관한 처리 추가, NULL 값은 변경되지 않도록 처리)
-2026-01-28	날씨 데이터 수집 수정(DB 및 java 코드상에 timezone 설정 추가)
-2026-01-28	데이터 품질(결측/이상치) 처리 완료
-2026-01-29	https를 위한 인증서 발급 완료
-2026-01-29	Oauth2 기능 추가 (구글, 네이버, 카카오)
-2026-01-29	Oauth2 테스트를 위한 ngrok 설치 및 테스트
-2026-01-29	날씨 데이터는 수정만 가능하도록함.
-2026-01-29	control => control과 service로 분리
-2026-01-30	Member 권한 : Role에 VIEW 권한 추가
-2026-01-30	API 요청 기록 저장
-2026-01-30	개발용 NextJS와 Springboot 간의 통신을 위해서 CORS 전부 수용
-2026-01-30	CORS 수용 리스트를 application.properties에서 조회하도록 처리
-2026-01-30	리눅스 설정 관련 정리
-2026-01-31	메모 관련 처리 추가
-2026-02-02	TMS 데이터를 csv로부터 추가
		api를 통해서 입력되는데 514552개중에 426520개를 저장하는데 90분 이상 걸림
		3000개 단위로 나눠서 처리
		JPA를 통한 saveAll을 대신해서 PreparedStatement를 통한 insert 쿼리를 구성해서 실행시 1분 이내
		다만 DB에 이미 저장된 값이 있는지 확인하였을때는 2분 정도의 시간이 소비됨
-2026-02-02	Member API 명칭 변경
-2026-02-02	swagger UI 용 어노테이션 추가 정리
-2026-02-02	로그인 실패 이력 처리 추가
-2026-02-03	*.projectwwtp.kro.kr 도메인에 대한 인증서 발급
-2026-02-03	메모 삭제 API 추가
-2026-02-03	Email 보내기 설정 및 API 추가
		- Admin, Member 들에게 한꺼번에 가도록 수정
-2026-02-03	회원 데이터에 Email 추가
-2026-02-03	TMS 원본 데이터에서 결측치를 추가하는 API 구성중
-2026-02-04	TMS 결측치 보정 후 CSV 파일로 저장
		CSV 파일이 존재 하면 해당 파일로부터 데이터를 불러오도록 처리
		API 응답 시간이 1.19 -> 0.32로 빨라짐
		CSV를 DB 저장으로 바꾸면 ->0.21로 빨라짐
-2026-02-04	회원 추가시 email 정규식 확인 추가
-2026-02-04	application.properties의 설정값을 통한 OAuth2 인증 처리 추가
		OAuth2에 대한 처리 방식에 URI 구성(/api/auth2**) 변경
-2026-02-04	TMS 데이터 관련 로그 추가
-2026-02-05	TMS 데이터 처리 방식 변경
		- 조회시점을 기준으로 이전 24시간의 데이터(1440개)를 조회
		- 실시간 처럼 보이기 위해서 데이터가 충실한(전날부터의 개수가 2600~2880) 날짜를 임의로 선정
		- 날짜 선정 및 결측/이상치 처리는 새벽 0시 5분, 서버 실행시 체크하도록 처리
-2026-02-05	파이썬에서 사용할 TMS 데이터와 AWS 데이터 구성
-2026-02-05	서버로의 잘못된 접근시 401, 403 일때의 JSON 구성(원치 않는 화면으로 자동적으로 넘어가지 않도록)
-2026-02-05	ASW 데이터 수집 오류 수정(데이터 재검증)
		수집 기준을 data_no가 아닌 time을 기준으로 처리
		수집과 검증의 스케쥴링 타이밍 별도 관리