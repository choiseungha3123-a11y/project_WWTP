package kr.kro.prjectwwtp.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
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
import kr.kro.prjectwwtp.config.TokenBlacklistManager;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.LoginLogService;
import kr.kro.prjectwwtp.service.MailService;
import kr.kro.prjectwwtp.service.MemberService;
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
	private final LoginLogService logService;
	private final TokenBlacklistManager tokenBlacklistManager;
	private final MailService mailService;
	
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
	
	@GetMapping("/health")
	public ResponseEntity<Object> healthCheck() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		return ResponseEntity.ok().body(res);
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
			
			// 기존 토큰 무효화 (다른 기기에서의 로그인을 무효화)
			tokenBlacklistManager.invalidateToken(req.userId);
			
			// 토큰 생성
			String token = JWTUtil.getJWT(member);
			//System.out.println("token : " + token);
			
			// 새 토큰 등록
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			
			tokenBlacklistManager.registerNewToken(req.userId, token, userAgent, remoteInfo);
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
			
			// TokenBlacklistManager에서 토큰 무효화
			tokenBlacklistManager.invalidateToken(userid);
			
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
		
		String domain = "http://10.125.121.176:8081";
		String subject = "Email 인증 From FlowWater"; 
		String link = domain + "/api/member/validateKey?keyValue="+key;
		String body = "<div style=\"font-family: 'Apple SD Gothic Neo', 'sans-serif' !important; width: 540px; height: 600px; border-top: 4px solid #3498db; margin: 100px auto; padding: 30px 0; box-sizing: border-box;\">" +
	              "    <h1 style=\"margin: 0; padding: 0 5px; font-size: 28px; font-weight: 400;\">" +
	              "        <span style=\"color: #3498db;\">" + subject + "</span> 안내" +
	              "    </h1>" +
	              "    <p style=\"font-size: 16px; line-height: 26px; margin-top: 50px; padding: 0 5px;\">" +
	              "        아래 버튼을 클릭하여 인증을 완료해 주세요.<br>" +
	              "        본 메일은 <b>FlowWater</b> 서비스 이용을 위해 발송되었습니다." +
	              "        본 메일의 인증은 10분 간만 유효합니다." +
	              "    </p>" +
	              "    <a href=\"" + link + "\" style=\"display: inline-block; width: 210px; height: 45px; margin: 30px 5px 40px; background: #3498db; color: #ffffff; text-decoration: none; text-align: center; line-height: 45px; vertical-align: middle; font-size: 16px; border-radius: 5px;\" target=\"_blank\">인증 완료하기</a>" +
	              "    <div style=\"border-top: 1px solid #DDD; padding: 5px;\">" +
	              "        <p style=\"font-size: 13px; line-height: 21px; color: #555;\">" +
	              "            만약 버튼이 작동하지 않는다면 아래 링크를 복사하여 브라우저에 붙여넣어 주세요.<br>" +
	              "            <span style=\"color: #3498db;\">" + link + "</span>" +
	              "        </p>" +
				  "        <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
				  "            도움이 필요하시면 <a href=\"https://www.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
				  "        </p>" +
	              "    </div>" +
	              "</div>";
		
		mailService.sendEmail(validateMember.getUserEmail(), subject, body);
		
		return ResponseEntity.ok().body(res);
	}
	
	public String failMessage(String errorMsg) {
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
			    "            요청하신 이메일 인증이 아래와 같은 사유로 완료되지 않았습니다." +
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
	
	public String successMessage(Member member) {
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
		    "            이메일 인증이 성공적으로 완료되었습니다." +
		    "        </p>" +
		    "        <div style=\"background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin: 20px 5px; border: 1px dashed #3498db;\">" +
		    "            <p style=\"margin: 0; font-size: 15px; color: #333; text-align: center;\">" +
		    "                <b>\"FlowWater와 함께 깨끗하고 스마트한 시작을 함께하세요!\"</b>" +
		    "            </p>" +
		    "        </div>" +
		    "        <p style=\"font-size: 16px; line-height: 26px; padding: 0 5px;\">" +
		    "            아래 버튼을 눌러 메인 화면으로 이동해 보세요." +
		    "        </p>" +
		    "        <a href=\"" + mainLink + "\" style=\"display: inline-block; width: 210px; height: 45px; margin: 30px 5px 40px; background: #3498db; color: #ffffff; text-decoration: none; text-align: center; line-height: 45px; vertical-align: middle; font-size: 16px; border-radius: 5px; font-weight: bold;\" target=\"_blank\">FlowWater 시작하기</a>" +
		    "        <div style=\"border-top: 1px solid #DDD; padding: 15px 5px;\">" +
		    "            <p style=\"font-size: 12px; line-height: 21px; color: #777; margin: 0;\">" +
		    "                도움이 필요하시면 <a href=\"http://wwws.projectwwtp.kro.kr/support\" style=\"color: #3498db; text-decoration: none;\">고객지원</a>으로 문의 바랍니다." +
		    "            </p>" +
		    "        </div>" +
		    "    </div>" +
		    "</body>" +
		    "</html>";
		return body;
	}
	
	@GetMapping("/validateKey")
	@Operation(summary="Email 인증 완료", description = "Email 인증 완료")
	@Parameter(name = "keyValue", description= "자동 발급된 인증키")
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public String postValidateKey(
			@RequestParam String keyValue) {
		String errorMsg = null;
		if(Util.isExpired(keyValue)) {
			errorMsg = "토큰이 만료되었습니다.";
			return failMessage(errorMsg);
		}
		Long userNo = Util.pareKey(keyValue);
		if(userNo < 0) {
			errorMsg = "토큰 정보가 올바르지 않습니다.";
			return failMessage(errorMsg);
		}
		Member member = memberService.findByNo(userNo);
		if(member.getValidateKey() == null ||!member.getValidateKey().equals(keyValue)) {
			errorMsg = "토큰 정보가 올바르지 않습니다.";
			return failMessage(errorMsg);
		}
		
		memberService.validEmail(userNo);;
		
		return successMessage(member);
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
