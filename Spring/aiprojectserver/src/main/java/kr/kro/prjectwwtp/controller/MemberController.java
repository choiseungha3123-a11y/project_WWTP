package kr.kro.prjectwwtp.controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.config.CryptoStringConverter;
import kr.kro.prjectwwtp.domain.FlowPredict;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsPredict;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.FlowService;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.service.MailService;
import kr.kro.prjectwwtp.service.MemberService;
import kr.kro.prjectwwtp.service.TmsService;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@RestController
@RestControllerAdvice
@RequestMapping("/api/member")
@RequiredArgsConstructor
@Tag(name="MemberController", description = "회원정보 관리 API")
public class MemberController {
	private final MemberService memberService;
	private final LogService logService;
	private final TmsService tmsService;
	private final FlowService flowService;
	private final MailService mailService;
	
	@Value("${report.enable}")
	private boolean enableReport;
	
	@ExceptionHandler(MissingServletRequestParameterException.class)
	public ResponseEntity<Object> handleMissingParams(MissingServletRequestParameterException ex) {
		responseDTO res = responseDTO.builder()
				.success(false)
				.errorMsg(ex.getParameterName() + " 파라메터가 누락되었습니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@ExceptionHandler(MethodArgumentTypeMismatchException.class)
	public ResponseEntity<Object> handleMismatchParams(MethodArgumentTypeMismatchException ex) {
		responseDTO res = responseDTO.builder()
				.success(false)
				.errorMsg(ex.getName() + " 파라메터의 형식이 올바르지 않습니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@ExceptionHandler(HttpRequestMethodNotSupportedException.class)
	public ResponseEntity<Object> handleMethodNotSupported(HttpRequestMethodNotSupportedException ext) {
		responseDTO res = responseDTO.builder()
				.success(false)
				.errorMsg(" 허용되지 않는 Method 입니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/test")
	public String test(@RequestParam String str) {
		CryptoStringConverter conv = new CryptoStringConverter();
		String enc = conv.convertToDatabaseColumn(str);
		String dec = conv.convertToEntityAttribute(enc);
		String ret = "orgin : " + str + "\n";
		ret += "enc : " + enc + "\n";
		ret += "dec : " + dec + "\n";
		return ret;
	}
	
	@Getter
	@Setter
	@ToString
	static public class memberLoginDTO {
		@Schema(name = "userId", description = "등록된 사용자 ID", example = "member")
		private String userId;
		@Schema(name = "password", description = "비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.", example = "member1234")
		private String password;
	}
	
	@PostMapping("/login")
	@Operation(summary="로그인 시도", description = "userid/password를 통해 로그인을 시도")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memberLoginDTO.class))
	@ApiResponse(description = "dataList[0]에 jwtToken을 사용해야합니다.", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> login(
			HttpServletRequest request,
			@RequestBody memberLoginDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		Member member = null;
		boolean loginSuccess = false;
		String userId = req.userId;
		
		String remoteInfo = null;
		String errorMsg = null;
		
		try {
			String remoteAddr = Util.getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			remoteInfo = remoteAddr + ":" + remotePort;
			if(req.userId == null || req.userId.length() == 0 
					|| req.password == null || req.password.length() == 0) {
				res.setSuccess(false);
				errorMsg = "정보가 올바르지 않습니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			//System.out.println("req : " + req);
			
			member = memberService.getByIdAndPassword(req.userId, req.password);
			
			if(member == null) {
				res.setSuccess(false);
				errorMsg = "회원 정보가 존재하지 않습니다. ID와 비밀번호를 확인해주세요.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			
			// 토큰 생성
			String token = JWTUtil.getJWT(member);
			//System.out.println("token : " + token);
			
			// 새 토큰 등록
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			
			loginSuccess = true;
			res.addData(token);
		}catch (Exception e) {
			res.setSuccess(false);
			errorMsg = e.getMessage();
		}finally {
			// 접속 로그 기록
			logService.addLoginLog(member, loginSuccess, userId, remoteInfo, null, errorMsg);
		}
		return ResponseEntity.ok().body(res);
		
	}
	
	@PostMapping("/logout")
	@Operation(summary="로그아웃", description = "사용자 로그아웃 처리")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> logout(HttpServletRequest request) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		
		// 종료 로그 기록 : 필요
		
		// 토큰 추출 및 검증
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		// JWT에서 userid 추출
		try {
			String token = request.getHeader("Authorization");
			String userid = JWTUtil.getClaim(token, JWTUtil.useridClaim);
			System.out.println("[MemberController] logout request for user: " + userid);
			
			res.setSuccess(true);
			res.setErrorMsg(null);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg("로그아웃 처리 중 오류가 발생했습니다.");
			System.out.println("[MemberController] logout error: " + e.getMessage());
		}
		
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/list")
	@Operation(summary="맴버 리스트 조회", description = "등록된 맴버 전체 리스트 조회")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = Member.class)))
	public ResponseEntity<Object> listMember(
			HttpServletRequest request) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		//String token = request.getHeader("Authorization");
		//System.out.println("token : " + token);
		Member member = JWTUtil.parseToken(request);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN){
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		List<Member> list = memberService.getMemberList();
		for(Member mem : list)
			res.addData(mem);
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/checkId")
	@Operation(summary="ID 중복 체크", description = "ID 중복체크")
	@Parameter(name = "userId", description = "확인할 사용자 ID")
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> checkId(@RequestParam String userId) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(userId == null || userId.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		if(memberService.checkId(userId)) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 ID 입니다.");
			return ResponseEntity.ok().body(res);
		}
		
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/checkEmail")
	@Operation(summary="Email 중복 체크", description = "Email 중복체크")
	@Parameter(name = "userEmail", description = "확인할 사용자 Email")
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> checkEmail(@RequestParam String userEmail) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(userEmail == null || userEmail.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		if(memberService.checkEmail(userEmail)) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 Email 입니다.");
			return ResponseEntity.ok().body(res);
		}
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class validEmailDTO {
		@Schema(description = "고유번호", example = "1~")
		private long userNo;
	}
	
	@Value("${spring.EmailAPI.URI}")
	private String emailAPIDomain;
	
	@PostMapping("/validateEmail")
	@Operation(summary="Email 인증 수행", description = "Email 인증 수행")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = validEmailDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postValidateEmail(
			HttpServletRequest request,
			@RequestBody validEmailDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member member = JWTUtil.parseToken(request);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN){
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member validateMember = memberService.findByNo(req.userNo);
		if(validateMember == null || validateMember.getRole() == Role.ROLE_VIEWER ) {
			res.setSuccess(false);
			res.setErrorMsg("인증하려는 사용자가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		String key = Util.getTempKey(validateMember.getUserNo());
		System.out.println("key : " + key);
		
		memberService.addEmailKey(validateMember.getUserNo(), key);
		String userId = validateMember.getUserId();
		String email = validateMember.getUserEmail();
		String subject = "Email 인증 From FlowWater"; 
		String validateLink = emailAPIDomain + "/api/member/validateKey?keyValue="+key;
		String deleteLink = emailAPIDomain + "/api/member/deleteEmail?userId="+userId+"&email="+email;
		String body = "<div style=\"font-family: 'Apple SD Gothic Neo', 'sans-serif' !important; width: 540px; height: 600px; border-top: 4px solid #3498db; margin: 100px auto; padding: 30px 0; box-sizing: border-box;\">" +
	              "    <h1 style=\"margin: 0; padding: 0 5px; font-size: 28px; font-weight: 400;\">" +
	              "        <span style=\"color: #3498db;\">" + subject + "</span> 안내" +
	              "    </h1>" +
	              "    <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
	              "        아래 버튼을 클릭하여 인증을 완료해 주세요.<br>" +
	              "        본 메일은 <b>FlowWater</b> 서비스 이용을 위해 발송되었습니다.<br>" +
	              "        본 메일의 인증은 10분 간만 유효합니다." +
	              "    </p>" +
	              "    <a href=\"" + validateLink + "\" style=\"display: inline-block; width: 210px; height: 45px; margin: 30px 5px 40px; background: #3498db; color: #ffffff; text-decoration: none; text-align: center; line-height: 45px; vertical-align: middle; font-size: 16px; border-radius: 5px;\" target=\"_blank\">인증 완료하기</a>" +
	              "    <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
	              "        더이상 이 보고서를 받지 않으시려면<br>" +
	              "        아래 버튼을 눌러 이메일 정보를 삭제하십시오..<br>" +
	              "    </p>" +
	              "    <a href=\"" + deleteLink + "\" style=\"display: inline-block; width: 210px; height: 45px; margin: 30px 5px 40px; background: #3498db; color: #ffffff; text-decoration: none; text-align: center; line-height: 45px; vertical-align: middle; font-size: 16px; border-radius: 5px;\" target=\"_blank\">이메일정보 삭제</a>" +
	              "    <div style=\"border-top: 1px solid #DDD; padding: 5px;\">" +
//	              "        <p style=\"font-size: 13px; line-height: 21px; color: #555;\">" +
//	              "            만약 버튼이 작동하지 않는다면 아래 링크를 복사하여 브라우저에 붙여넣어 주세요.<br>" +
//	              "            <span style=\"color: #3498db;\">" + validateLink + "</span>" +
//	              "        </p>" +
				  "        <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
				  "            도움이 필요하시면 <a href=\"https://www.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
				  "        </p>" +
	              "    </div>" +
	              "</div>";
		
		mailService.sendEmail(validateMember.getUserEmail(), subject, body);
		
		return ResponseEntity.ok().body(res);
	}
	
	public String failMessage(String type, String errorMsg) {
		String titleText = "FlowWater 인증 실패 안내";
		String body = "<!DOCTYPE html>" +
			    "<html>" +
			    "<head>" +
			    "    <meta charset=\"UTF-8\">" +
			    "    <title>" + titleText + "</title>" + // 브라우저 탭 타이틀
			    "</head>" +
			    "<body style=\"margin: 0; padding: 0;\">" +
			    "    <div style=\"font-family: 'Apple SD Gothic Neo', 'sans-serif' !important; width: 540px; border-top: 4px solid #e74c3c; margin: 50px auto; padding: 30px 0; box-sizing: border-box;\">" +
			    "        <h1 style=\"margin: 0; padding: 0 5px; font-size: 28px; font-weight: 400;\">" +
			    "            <span style=\"color: #e74c3c;\">인증 실패</span> 안내" +
			    "        </h1>" +
			    "        <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
			    "            안녕하세요, <b>FlowWater</b>입니다.<br>" +
			    "            요청하신 <b>"+ type + "</b>이 아래와 같은 사유로 완료되지 않았습니다." +
			    "        </p>" +
			    "        <div style=\"background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 5px;\">" +
			    "            <p style=\"margin: 0; font-size: 15px; color: #333;\">" +
			    "                <b>실패 사유:</b> <span style=\"color: #e74c3c;\">" + errorMsg + "</span>" +
			    "            </p>" +
			    "        </div>" +
			    "        <div style=\"border-top: 1px solid #DDD; padding: 15px 5px;\">" +
			    "            <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
			    "                도움이 필요하시면 <a href=\"https://www.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
			    "            </p>" +
			    "        </div>" +
			    "    </div>" +
			    "</body>" +
			    "</html>";
		return body;
	}
	
	public String successMessage(String type, Member member) {
		String titleText = "FlowWater 가입을 환영합니다!";
		String mainLink = "https://www.projectwwtp.kro.kr";

		String body = 
		    "<!DOCTYPE html>" +
		    "<html>" +
		    "<head>" +
		    "    <meta charset=\"UTF-8\">" +
		    "    <title>" + titleText + "</title>" +
		    "</head>" +
		    "<body style=\"margin: 0; padding: 0;\">" +
		    "    <div style=\"font-family: 'Apple SD Gothic Neo', 'sans-serif' !important; width: 540px; border-top: 4px solid #3498db; margin: 50px auto; padding: 30px 0; box-sizing: border-box;\">" +
		    "        <h1 style=\"margin: 0; padding: 0 5px; font-size: 28px; font-weight: 400;\">" +
		    "            <span style=\"color: #3498db;\">인증 성공!</span> 환영합니다" +
		    "        </h1>" +
		    "        <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
		    "            안녕하세요, <b>" + member.getUserName() + "</b>님!<br>" +
		    "            <b>"+ type + "</b>이 성공적으로 완료되었습니다." +
		    "        </p>";
	    	if(type.equals("이메일 인증")) {		    
	    		body += "        <div style=\"background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin: 20px 5px; border: 1px dashed #3498db;\">" +
		    "            <p style=\"margin: 0; font-size: 15px; color: #333; text-align: center;\">" +
		    "                <b>\"FlowWater와 함께 깨끗하고 스마트한 시작을 함께하세요!\"</b>" +
		    "            </p>" +
		    "        </div>" +
		    "        <p style=\"font-size: 16px; line-height: 26px; padding: 0 5px;\">" +
		    "            아래 버튼을 눌러 메인 화면으로 이동해 보세요." +
		    "        </p>" +
		    "        <a href=\"" + mainLink + "\" style=\"display: inline-block; width: 210px; height: 45px; margin: 30px 5px 40px; background: #3498db; color: #ffffff; text-decoration: none; text-align: center; line-height: 45px; vertical-align: middle; font-size: 16px; border-radius: 5px; font-weight: bold;\" target=\"_blank\">FlowWater 시작하기</a>";
	    	}
		    body += "        <div style=\"border-top: 1px solid #DDD; padding: 15px 5px;\">" +
		    "            <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
		    "                도움이 필요하시면 <a href=\"http://wwws.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
		    "            </p>" +
		    "        </div>" +
		    "    </div>" +
		    "</body>" +
		    "</html>";
		return body;
	}
	

	
	boolean bSendEmail = true;
	@Scheduled(cron = "${scheduler.report.cron}")
	@GetMapping("/mailtest")
	public void makeReportMessage()
	{
		if(!enableReport) return;
		LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
		LocalDateTime end = now.plusDays(1).minusMinutes(1);
		List<TmsPredict> tmsList = tmsService.findPredictList(now, end);
		List<FlowPredict> flowList = flowService.findPredictList(now, end);
		
		String subject = "Report From FlowWater";
		String titleText = "FlowWater Report";
		String nowStr = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		int angle = 0;
		
		String body = "<div style=\"font-family: 'Apple SD Gothic Neo', 'sans-serif' !important; width: 540px; height: 600px; border-top: 4px solid #3498db; margin: 100px auto; padding: 30px 0; box-sizing: border-box;\">" +
	              "    <h1 style=\"margin: 0; padding: 0 5px; font-size: 28px; font-weight: 400;\">" +
	              "        <span style=\"color: #3498db;\">" + subject + "</span> 안내" +
	              "    </h1>" +
	              "    <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
	              "        첨부된 파일을 다운 받아 12시간 동안의 예측차트를 확인해보세요.<br>" +
	              "    </p>" +
	              "    <div style=\"border-top: 1px solid #DDD; padding: 5px;\">" +
				  "        <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
				  "            도움이 필요하시면 <a href=\"https://www.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
				  "        </p>" +
	              "    </div>" +
	              "</div>";
		
		String html = "<!DOCTYPE html>\r\n"
				+ "        <html>\r\n"
				+ "        <head>\r\n"
				+ "            <meta charset=\"UTF-8\">\r\n"
				+ "            <style>\r\n"
				+ "                body { font-family: 'Malgun Gothic', sans-serif; background: #f4f7f9; padding: 20px; color: #333; }\r\n"
				+ "                .container { max-width: 1000px; margin: auto; }\r\n"
				+ "                \r\n"
				+ "                /* 상단 작성 시간 스타일 */\r\n"
				+ "                .timestamp { text-align: right; font-size: 14px; color: #666; margin-bottom: 10px; font-weight: bold; }\r\n"
				+ "                \r\n"
				+ "                .chart-card { background: white; margin-bottom: 25px; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }\r\n"
				+ "                h2 { font-size: 1.1rem; margin-bottom: 15px; border-left: 5px solid #3498db; padding-left: 10px; color: #2c3e50; }\r\n"
				+ "                \r\n"
				+ "                .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; padding: 10px; background: #fafafa; border-radius: 6px; }\r\n"
				+ "                .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; font-weight: bold; }\r\n"
				+ "                .legend-color { width: 10px; height: 10px; border-radius: 2px; }\r\n"
				+ "                \r\n"
				+ "                svg { width: 100%; height: auto; display: block; }\r\n"
				+ "                .axis { stroke: #ccc; stroke-width: 1; }\r\n"
				+ "                .grid { stroke: #f0f0f0; stroke-width: 1; }\r\n"
				+ "                .line { fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }\r\n"
				+ "                .label { font-size: 10px; fill: #888; }\r\n"
				+ "                .point { fill: white; stroke-width: 1.5; cursor: pointer; transition: r 0.2s; }\r\n"
				+ "                .point:hover { r: 5; }\r\n"
				+ "                \r\n"
				+ "                /* 툴팁 스타일 */\r\n"
				+ "                .tooltip {\r\n"
				+ "                    position: absolute;\r\n"
				+ "                    background: rgba(0, 0, 0, 0.85);\r\n"
				+ "                    color: white;\r\n"
				+ "                    padding: 8px 12px;\r\n"
				+ "                    border-radius: 6px;\r\n"
				+ "                    font-size: 12px;\r\n"
				+ "                    pointer-events: none;\r\n"
				+ "                    opacity: 0;\r\n"
				+ "                    transition: opacity 0.2s;\r\n"
				+ "                    z-index: 1000;\r\n"
				+ "                    white-space: nowrap;\r\n"
				+ "                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);\r\n"
				+ "                }\r\n"
				+ "                .tooltip.show {\r\n"
				+ "                    opacity: 1;\r\n"
				+ "                }\r\n"
				+ "                .tooltip-time {\r\n"
				+ "                    font-weight: bold;\r\n"
				+ "                    margin-bottom: 4px;\r\n"
				+ "                    color: #3498db;\r\n"
				+ "                }\r\n"
				+ "                .tooltip-value {\r\n"
				+ "                    margin: 2px 0;\r\n"
				+ "                }\r\n"
				+ "            </style>"
				+ "    			<title>" + titleText + "</title>" + // 브라우저 탭 타이틀
				"        </head>\r\n"
				+ "        <body>\r\n"
				+ "            <div class=\"container\">\r\n"
				+ "                <!-- 작성 시간 표시 영역 -->\r\n"
				+ "                <div class=\"timestamp\" id=\"current-time\">작성 시간 : </div>\r\n"
				+ "\r\n"
				+ "                <!-- 차트 1: 유입유량 -->\r\n"
				+ "                <div class=\"chart-card\">\r\n"
				+ "                    <h2>유입유량 예측</h2>\r\n"
				+ "                    <div id=\"chart-flow\" style=\"position: relative;\"></div>\r\n"
				+ "                </div>\r\n"
				+ "\r\n"
				+ "                <!-- 차트 2: TMS 통합 정보 -->\r\n"
				+ "                <div class=\"chart-card\">\r\n"
				+ "                    <h2>TMS 예측</h2>\r\n"
				+ "                    <div class=\"legend\" id=\"tms-legend\"></div>\r\n"
				+ "                    <div id=\"chart-tms\" style=\"position: relative;\"></div>\r\n"
				+ "                </div>\r\n"
				+ "            </div>\r\n"
				+ "\r\n"
				+ "            <!-- 툴팁 엘리먼트 -->\r\n"
				+ "            <div class=\"tooltip\" id=\"tooltip\"></div>\r\n"
				+ "\r\n"
				+ "            <script>\r\n"
				+ "                // 현재 시간 표시 함수\r\n"
				+ "                function updateTimestamp() {\r\n"
				+ "                    document.getElementById('current-time').innerText = \"작성 시간 : " + nowStr + "\"\r\n"
				+ "				}\r\n"
				+ "				updateTimestamp();\r\n"
				+ "				\r\n"
				+ "				// 1. 데이터 정의\r\n"
				+ "				const flowData = [\r\n";
		for(int i = 0; i < flowList.size(); ++i) {
			FlowPredict flow = flowList.get(i);
			html += "{ time: \"" + flow.getFlowTime().format(DateTimeFormatter.ofPattern("HH:mm")) + "\", val: " + flow.getFlowValue() + "}";
			if (i < flowList.size() - 1)
				html += ",\r\n";
		}
		html += "                ];\r\n" +
				"\r\n" +
				"                const tmsData = [\r\n";
		for(int i = 0; i < tmsList.size(); ++i) {
			TmsPredict tms = tmsList.get(i);
			html += "{ time: \"" + tms.getTmsTime().format(DateTimeFormatter.ofPattern("HH:mm")) + 
					"\", TOC: " + tms.getToc() + 
					", PH: " + tms.getPh() +
					", SS: " + tms.getSs() +
					", FLUS: " + tms.getFlux() +
					", TN: " + tms.getTn() +
					", TP: " + tms.getTp() + "}";
			if (i < tmsList.size() - 1)
				html += ",\r\n";
		}
		html += "                ];\r\n"
				+ "\r\n"
				+ "                const tmsKeys = [\"TOC\", \"PH\", \"SS\", \"FLUS\", \"TN\", \"TP\"];\r\n"
				+ "                const colors = {\r\n"
				+ "                    TOC: \"#e74c3c\", PH: \"#2ecc71\", SS: \"#f1c40f\", \r\n"
				+ "                    FLUS: \"#9b59b6\", TN: \"#34495e\", TP: \"#e67e22\", flow: \"#3498db\"\r\n"
				+ "                };\r\n"
				+ "\r\n"
				+ "                // 범례 생성\r\n"
				+ "                const legendBox = document.getElementById('tms-legend');\r\n"
				+ "                tmsKeys.forEach(key => {\r\n"
				+ "                    legendBox.innerHTML += `<div class=\"legend-item\"><div class=\"legend-color\" style=\"background:${colors[key]}\"></div>${key}</div>`;\r\n"
				+ "                });\r\n"
				+ "\r\n"
				+ "                // 툴팁 엘리먼트\r\n"
				+ "                const tooltip = document.getElementById('tooltip');\r\n"
				+ "\r\n"
				+ "                // 툴팁 표시 함수\r\n"
				+ "                function showTooltip(event, data, key, isSingle) {\r\n"
				+ "                    const value = isSingle ? data.val : data[key];\r\n"
				+ "                    const formattedValue = value.toFixed(2);\r\n"
				+ "                    \r\n"
				+ "                    let content = `<div class=\"tooltip-time\">${data.time}</div>`;\r\n"
				+ "                    if (isSingle) {\r\n"
				+ "                        content += `<div class=\"tooltip-value\">유량: ${formattedValue}</div>`;\r\n"
				+ "                    } else {\r\n"
				+ "                        content += `<div class=\"tooltip-value\">${key}: ${formattedValue}</div>`;\r\n"
				+ "                    }\r\n"
				+ "                    \r\n"
				+ "                    tooltip.innerHTML = content;\r\n"
				+ "                    tooltip.classList.add('show');\r\n"
				+ "                    \r\n"
				+ "                    // 툴팁 위치 설정\r\n"
				+ "                    const x = event.pageX + 10;\r\n"
				+ "                    const y = event.pageY - 10;\r\n"
				+ "                    tooltip.style.left = x + 'px';\r\n"
				+ "                    tooltip.style.top = y + 'px';\r\n"
				+ "                }\r\n"
				+ "\r\n"
				+ "                // 툴팁 숨김 함수\r\n"
				+ "                function hideTooltip() {\r\n"
				+ "                    tooltip.classList.remove('show');\r\n"
				+ "                }\r\n"
				+ "\r\n"
				+ "                function drawChart(targetId, data, keys, isSingle) {\r\n"
				+ "                    const width = 800, height = 300, padding = 40;\r\n"
				+ "                    const chartW = width - (padding * 2);\r\n"
				+ "                    const chartH = height - (padding * 2);\r\n"
				+ "\r\n"
				+ "                    let svg = `<svg viewBox=\"0 0 ${width} ${height}\" xmlns=\"http://www.w3.org/2000/svg\">`;\r\n"
				+ "                    \r\n"
				+ "                    data.forEach((d, i) => {\r\n"
				+ "                        let x = padding + (i * (chartW / (data.length - 1)));\r\n"
				+ "                        svg += `<line x1=\"${x}\" y1=\"${padding}\" x2=\"${x}\" y2=\"${height-padding}\" class=\"grid\" />`;\r\n"
				+ "                        svg += `<text x=\"${x}\" y=\"${height-padding+20}\" class=\"label\" text-anchor=\"middle\" transform=\"rotate(" + angle + " ${x} ${height-padding+20})\">${d.time}</text>`;\r\n"
				+ "                    });\r\n"
				+ "\r\n"
				+ "                    keys.forEach(key => {\r\n"
				+ "                        const vals = data.map(d => isSingle ? d.val : d[key]);\r\n"
				+ "                        const min = Math.min(...vals), max = Math.max(...vals);\r\n"
				+ "                        const range = (max - min === 0) ? 1 : (max - min);\r\n"
				+ "\r\n"
				+ "                        let points = data.map((d, i) => {\r\n"
				+ "                            let val = isSingle ? d.val : d[key];\r\n"
				+ "                            let x = padding + (i * (chartW / (data.length - 1)));\r\n"
				+ "                            let y = height - padding - ((val - min) / range * chartH);\r\n"
				+ "                            return `${x},${y}`;\r\n"
				+ "                        }).join(\" \");\r\n"
				+ "\r\n"
				+ "                        let strokeColor = isSingle ? colors.flow : colors[key];\r\n"
				+ "                        svg += `<polyline points=\"${points}\" class=\"line\" stroke=\"${strokeColor}\" />`;\r\n"
				+ "                        \r\n"
				+ "                        data.forEach((d, i) => {\r\n"
				+ "                            let val = isSingle ? d.val : d[key];\r\n"
				+ "                            let x = padding + (i * (chartW / (data.length - 1)));\r\n"
				+ "                            let y = height - padding - ((val - min) / range * chartH);\r\n"
				+ "                            svg += `<circle cx=\"${x}\" cy=\"${y}\" r=\"3\" class=\"point\" stroke=\"${strokeColor}\" data-index=\"${i}\" data-key=\"${key}\" />`;\r\n"
				+ "                        });\r\n"
				+ "                    });\r\n"
				+ "\r\n"
				+ "                    svg += `</svg>`;\r\n"
				+ "                    document.getElementById(targetId).innerHTML = svg;\r\n"
				+ "\r\n"
				+ "                    // SVG에 이벤트 리스너 추가\r\n"
				+ "                    const svgElement = document.querySelector(`#${targetId} svg`);\r\n"
				+ "                    svgElement.addEventListener('mouseover', function(e) {\r\n"
				+ "                        if (e.target.classList.contains('point')) {\r\n"
				+ "                            const index = parseInt(e.target.getAttribute('data-index'));\r\n"
				+ "                            const key = e.target.getAttribute('data-key');\r\n"
				+ "                            showTooltip(e, data[index], key, isSingle);\r\n"
				+ "                        }\r\n"
				+ "                    });\r\n"
				+ "\r\n"
				+ "                    svgElement.addEventListener('mouseout', function(e) {\r\n"
				+ "                        if (e.target.classList.contains('point')) {\r\n"
				+ "                            hideTooltip();\r\n"
				+ "                        }\r\n"
				+ "                    });\r\n"
				+ "\r\n"
				+ "                    svgElement.addEventListener('mousemove', function(e) {\r\n"
				+ "                        if (e.target.classList.contains('point')) {\r\n"
				+ "                            const x = e.pageX + 10;\r\n"
				+ "                            const y = e.pageY - 10;\r\n"
				+ "                            tooltip.style.left = x + 'px';\r\n"
				+ "                            tooltip.style.top = y + 'px';\r\n"
				+ "                        }\r\n"
				+ "                    });\r\n"
				+ "                }\r\n"
				+ "\r\n"
				+ "                drawChart('chart-flow', flowData, ['flow'], true);\r\n"
				+ "                drawChart('chart-tms', tmsData, tmsKeys, false);\r\n"
				+ "            </script>\r\n"
			    + "        <div style=\"border-top: 1px solid #DDD; padding: 15px 5px;\">"
			    + "            <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">"
			    + "                도움이 필요하시면 <a href=\"https://www.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다."
			    + "            </p>"
			    + "        </div>"
				+ "        </body>\r\n"
				+ "        </html>";

		String fileName = "chart" + now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + ".html";

		try {
			if(bSendEmail) {
				List<String> emailList = memberService.getValidateEmail();
				mailService.sendEmailWithAttachment(emailList, subject, body, html, fileName);
			}
			else {
				saveChartFile(html, fileName);
			}
		}catch(Exception e) {
			
		}
		
	}
	
	private void saveChartFile(String body, String filepath) throws Exception {
		try {
			File file = Util.resolveFilePath(filepath);
			// 파일 경로의 디렉토리 생성
			File parentDir = file.getParentFile();
			if(parentDir != null && !parentDir.exists()) {
				boolean dirCreated = parentDir.mkdir();
				if(!dirCreated && !parentDir.exists()) {
					throw new Exception("디렉토리 생성 실패: " + parentDir.getAbsolutePath());
				}
				System.out.println("[saveChartFile] 디렉토리 생성: " + parentDir.getAbsolutePath());
			}
			
			// 부모 디렉토리 쓰기 권한 확인
			if (parentDir != null && !parentDir.canWrite()) {
				throw new Exception("디렉토리 쓰기 권한 없음: " + parentDir.getAbsolutePath());
			}
			
			// UTF-8 인코딩으로 CSV 파일 작성
			try (BufferedWriter bw = new BufferedWriter(
					new OutputStreamWriter(new FileOutputStream(file.getAbsolutePath()), "UTF-8"))) {
				bw.write(body);
			}
			
		} catch (Exception e) {
			System.err.println("[saveChartFile] 파일 저장 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			throw new Exception("파일 저장 중 오류가 발생했습니다: " + e.getMessage());
		}
		
	}
	
	@GetMapping("/validateKey")
	@Operation(summary="Email 인증 완료", description = "Email 인증 완료")
	@Parameter(name = "keyValue", description= "자동 발급된 인증키")
	@ApiResponse(description = "실패/ 성공 유무를 웹브라우져에서 보여줄 HTML 문서 형태로 처리")
	public String getValidateKey(
			@RequestParam String keyValue) {
		String errorMsg = null;
		if(Util.isExpired(keyValue)) {
			errorMsg = "토큰이 만료되었습니다.";
			return failMessage("이메일 인증", errorMsg);
		}
		Long userNo = Util.pareKey(keyValue);
		if(userNo < 0) {
			errorMsg = "토큰 정보가 올바르지 않습니다.";
			return failMessage("이메일 인증", errorMsg);
		}
		Member member = memberService.findByNo(userNo);
		if(member.getValidateKey() == null ||!member.getValidateKey().equals(keyValue)) {
			errorMsg = "토큰 정보가 올바르지 않습니다.";
			return failMessage("이메일 인증", errorMsg);
		}
		
		memberService.validEmail(userNo);;
		
		return successMessage("이메일 인증", member);
	}
	
	@GetMapping("/deleteEmail")
	@Operation(summary="Email 삭제", description = "Email 삭제")
	@Parameter(name = "userId", description = "확인할 사용자 ID")
	@Parameter(name = "email", description= "삭제할 Email 주소")
	@ApiResponse(description = "실패/ 성공 유무를 웹브라우져에서 보여줄 HTML 문서 형태로 처리")
	public String getDeleteEmail(
			@RequestParam String userId,
			@RequestParam String email) {
		String errorMsg = null;
		Member member = memberService.findById(userId);
		if(member.getUserEmail() == null || !member.getUserEmail().equals(email)) {
			errorMsg = "정보가 올바르지 않습니다.";
			return failMessage("이메일 삭제", errorMsg);
		}
		
		memberService.delteEmail(member.getUserNo());
		
		return successMessage("이메일 삭제", member);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memberCreateDTO {
		@Schema(name = "userId", description = "등록할 사용자 ID", example = "member")
		private String userId;
		@Schema(name = "password", description = "비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.", example = "member1234")
		private String password;
		@Schema(name = "userName", description = "등록할 사용자명", example = "member")
		private String userName;
		@Schema(name = "userEmail", description = "등록할 Email", example = "xxx@xxx.xom")
		private String userEmail;;
	}
	
	boolean validatePassword(String password) {
		return password.matches("^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{10,20}$");
	}
	
	boolean validateMail(String email) {
		return email.matches("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$");
	}
	
	@PutMapping("/create")
	@Operation(summary="맴버 추가", description = "userid/password/userName값을 맴버에 추가")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memberCreateDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> createMember(
			HttpServletRequest request,
			@RequestBody memberCreateDTO req
			) {
		//System.out.println("req : " + req);
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(req.userId == null || req.userId.length() == 0 
				|| req.password == null || req.password.length() == 0
				|| req.userName == null || req.userName.length() == 0
				|| req.userEmail == null || req.userEmail.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
//		if(!validatePassword(req.password))
//		{
//			res.setSuccess(false);
//			res.setErrorMsg("비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.");
//		}
		if(!validateMail(req.userEmail))
		{
			res.setSuccess(false);
			res.setErrorMsg("유효하지 않은 이메일 주소입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member member = JWTUtil.parseToken(request);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN){
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		if(memberService.checkId(req.userId)) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 ID 입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(memberService.checkEmail(req.userEmail)) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 Email 입니다.");
			return ResponseEntity.ok().body(res);
		}
		
		memberService.addMember(req.userId, req.password, req.userName, req.userEmail);
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memberModifyDTO {
		@Schema(name = "userNo", description = "등록된 사용자 고유번호", example = "1~")
		private long userNo;
		@Schema(name = "userId", description = "변경할 사용자 ID", example = "member")
		private String userId;
		@Schema(name = "password", description = "비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.", example = "member1234")
		private String password;
		@Schema(name = "userName", description = "변경할 사용자명", example = "member")
		private String userName;
		@Schema(name = "userEmail", description = "변경할 Email", example = "xxx@xxx.xxx")
		private String userEmail;
		@Schema(name = "role", description = "변경할 사용자 권한", example = "ROLE_VIEWER")
		private Role role;
	}
		
	@PatchMapping("/modify")
	@Operation(summary="맴버 정보 변경", description = "userNo를 이용해서 userId,password,role을 변경")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memberModifyDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> modifyMember(
			HttpServletRequest request,
			@RequestBody memberModifyDTO req
			) {
		//System.out.println("req : " + req);
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if((req.userId == null || req.userId.length() == 0) 
				&& (req.password == null || req.password.length() == 0)
				&& (req.userName == null || req.userName.length() == 0)
				&& req.role == null) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		if(!validateMail(req.userEmail))
		{
			res.setSuccess(false);
			res.setErrorMsg("유효하지 않은 이메일 주소입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member member = JWTUtil.parseToken(request);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN && member.getUserNo() != req.userNo) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member modifyMember = memberService.findByNo(req.userNo);
		if(modifyMember == null) {
			res.setSuccess(false);
			res.setErrorMsg("존재하지 않는 회원정보입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getUserNo() == req.userNo) {
			// 자기 정보 수정시
			if(!validatePassword(req.password))
			{
				res.setSuccess(false);
				res.setErrorMsg("비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.");
				return ResponseEntity.ok().body(res);
			}
			if(!member.getUserId().equals(req.userId) 
					&& memberService.checkId(req.userId)) {
				res.setSuccess(false);
				res.setErrorMsg("이미 사용중인 ID 입니다.");
				return ResponseEntity.ok().body(res);
			}
		} else {
			// 관리자가 정보 수정시
			if(!modifyMember.getUserId().equals(req.userId) 
					&& memberService.checkId(req.userId)) {
				res.setSuccess(false);
				res.setErrorMsg("이미 사용중인 ID 입니다.");
				return ResponseEntity.ok().body(res);
			}
		}
		
		memberService.modifyMember(modifyMember, req.userId, req.password, req.userName, req.userEmail, req.role);
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memberDeleteDTO {
		@Schema(name = "userNo", description = "삭제할 사용자 고유번호", example = "1~")
		private long userNo;
		@Schema(name = "userId", description = "삭제할 사용자 ID", example = "member")
		private String userId;
	}
	
	@DeleteMapping("/delete")
	@Operation(summary="맴버 정보 삭제", description = "userNo/userId를 이용해서 회원정보를 삭제")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memberDeleteDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> deleteMember(
			HttpServletRequest request,
			@RequestBody memberDeleteDTO req) {
		//System.out.println("req : " + req);
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(req.userId == null || req.userId.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		if(JWTUtil.isExpired(request))
		{
			res.setSuccess(false);
			res.setErrorMsg("토큰이 만료되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member member = JWTUtil.parseToken(request);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		Member deleteMember = memberService.findByNo(req.userNo);
		if(deleteMember == null) {
			res.setSuccess(false);
			res.setErrorMsg("존재하지 않는 회원정보입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN && member.getUserNo() != req.userNo) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		memberService.deleteMember(deleteMember);
		//System.out.println("delete success");
		
		return ResponseEntity.ok().body(res);
	}
}
