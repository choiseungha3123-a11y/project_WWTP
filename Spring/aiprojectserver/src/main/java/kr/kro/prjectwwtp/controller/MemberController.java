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
import io.swagger.v3.oas.annotations.Parameters;
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
import kr.kro.prjectwwtp.service.MemberService;
import kr.kro.prjectwwtp.util.JWTUtil;
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
	
	@Getter
	@Setter
	@ToString
	static public class loginDTO {
		private String userId;
		private String password;
	}
	
	@PostMapping("/login")
	@Operation(summary="로그인 시도", description = "userid/password를 통해 로그인을 시도")
	@Parameters( {
		@Parameter(name = "userId", description= "등록된 사용자 ID"),
		@Parameter(name = "password", description= "등록된 비밀번호")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 1<br>dataList : JWTToken<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> login(
			@RequestBody loginDTO req,
			HttpServletRequest request) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(req.userId == null || req.userId.length() == 0 
				|| req.password == null || req.password.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		//System.out.println("req : " + req);
		
		Member member = memberService.getByIdAndPassword(req.userId, req.password);
		
		if(member == null) {
			res.setSuccess(false);
			res.setErrorMsg("회원 정보가 존재하지 않습니다. ID와 비밀번호를 확인해주세요.");
			return ResponseEntity.ok().body(res);
		}
		
		// 접속 로그 기록
		logService.addLog(member);
		
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
		String remoteAddr = getRemoteAddress(request);
		int remotePort = request.getRemotePort();
		String remoteInfo = remoteAddr + ":" + remotePort;
		
		tokenBlacklistManager.registerNewToken(req.userId, token, userAgent, remoteInfo);
		
		res.addData(token);
		return ResponseEntity.ok().body(res);
	}
	
	@PostMapping("/logout")
	@Operation(summary="로그아웃", description = "사용자 로그아웃 처리")
	@ApiResponse(description = "success : 성공/실패<br>errorMsg : 오류 메시지")
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
	
	@GetMapping("/listMember")
	@Operation(summary="맴버 리스트 조회", description = "등록된 맴버 전체 리스트 조회")
	@ApiResponse(description = "success : 성공/실패<br>dataSize : dataList에 들어 있는 값들의 개수<br>dataList : 결과값배열<br>errorMsg : success가 false 일때의 오류원인 ", content = @Content(schema = @Schema(implementation = Member.class)))
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
	
	@Getter
	@Setter
	@ToString
	static public class checkDTO {
		private String userId;
	}
	
	@GetMapping("/checkId")
	@Operation(summary="ID 중복 체크", description = "ID 중복체크")
	@Parameters( {
		@Parameter(name = "userId", description = "확인할 사용자 ID"),
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
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
	
	@Getter
	@Setter
	@ToString
	static public class addDTO {
		private String userId;
		private String password;
		private String userName;
	}
	
	boolean validatePassword(String password) {
		return password.matches("^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{10,20}$");
	}
	
	@PutMapping("/addMember")
	@Operation(summary="맴버 추가", description = "userid/password/role값을 맴버에 추가")
	@Parameters( {
		@Parameter(name = "userId", description = "등록할 사용자 ID"),
		@Parameter(name = "password", description = "등록할 비밀번호"),
		@Parameter(name = "userName", description = "등록할 유저명"),
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> addMember(
			HttpServletRequest request,
			@RequestBody addDTO req
			) {
		//System.out.println("req : " + req);
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(req.userId == null || req.userId.length() == 0 
				|| req.password == null || req.password.length() == 0
				|| req.userName == null || req.userName.length() == 0) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
//		if(!validatePassword(req.password))
//		{
//			res.setSuccess(false);
//			res.setErrorMsg("비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.");
//		}
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
		
		memberService.addMember(req.userId, req.password, req.userName);
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class modifyDTO {
		private long userNo;
		private String userId;
		private String password;
		private String userName;
		private Role role;
	}
		
	@PatchMapping("/modifyMember")
	@Operation(summary="맴버 정보 변경", description = "userNo를 이용해서 userId,password,role을 변경")
	@Parameters( {
		@Parameter(name = "userNo", description= "변경할 사용자 고유번호"),
		@Parameter(name = "userId", description= "변경할 사용자 ID"),
		@Parameter(name = "password", description= "변경할 비밀번호"),
		@Parameter(name = "userName", description = "변경할 유저명"),
		@Parameter(name = "role", description = "변경할 이용자 권한", example = "ROLE_MEMBER")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> modifyMember(
			HttpServletRequest request,
			@RequestBody modifyDTO req
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
		Member modifyMember = memberService.getByNo(req.userNo);
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
		
		memberService.modifyMember(modifyMember, req.userId, req.password, req.userName, req.role);
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class deleteDTO {
		private long userNo;
		private String userId;
	}
	
	@DeleteMapping("/deleteMember")
	@Operation(summary="맴버 정보 삭제", description = "userNo/userId를 이용해서 회원정보를 삭제")
	@Parameters( {
		@Parameter(name = "userNo", description= "변경할 사용자 고유번호"),
		@Parameter(name = "userId", description= "변경할 사용자 ID")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> deleteMember(
			HttpServletRequest request,
			@RequestBody deleteDTO req) {
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
		Member deleteMember = memberService.getByNo(req.userNo);
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
	
	/**
	 * 클라이언트의 실제 IP 주소 추출
	 * 프록시 환경에서도 올바른 IP를 가져오도록 처리
	 */
	private String getRemoteAddress(HttpServletRequest request) {
		String ip = request.getHeader("X-Forwarded-For");
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getHeader("Proxy-Client-IP");
		}
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getHeader("WL-Proxy-Client-IP");
		}
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getRemoteAddr();
		}
		// X-Forwarded-For가 여러 IP를 포함할 수 있으므로 첫 번째만 사용
		if (ip != null && ip.contains(",")) {
			ip = ip.split(",")[0].trim();
		}
		return ip;
	}
}
