DB URL : miniproject.c70cou4sk2dq.ap-northeast-2.rds.amazonaws.com:3306
DB Schema : aiproject
User : aiproject
Password : projectwwtp

실행중인 JAR 확인
ps -ef | grep jar


http://10.125.121.176:8081/swagger-ui/index.html




해야할 일 : 
	HTTP 인증 및 도메인 처리
	기상 데이터(강수량, 기온, 습도)를 DB로 저장
	 - 24년부터 가져오되... 1일 단위로 
	 - 현재 기준이 되면 30분(혹은 1시간) 단위로 변경
	서버상에 파이썬 세팅
	amazone linux 2023 let's encrypt
	vscode 안티그라비티
	메모 처리
	


완료한 일 :
	강의실에서 리눅스 서버 연결
	강의실에서 DB 서버 연결
	집에서 리눅스 서버 연결
	집에서 DB 서버 연결
	기상청 API 허브 가입 및 인증키 발급
	swagger UI를 통한 API 설명 페이지 구성
	로그인 토큰 처리 추가
	회원 관리(로그인/추가/변경/삭제)
	