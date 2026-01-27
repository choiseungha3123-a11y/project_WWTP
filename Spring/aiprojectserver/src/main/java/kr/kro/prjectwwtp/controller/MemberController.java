package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

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
import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.LoginLog;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.LoginLogRepository;
import kr.kro.prjectwwtp.persistence.MemberRepository;
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
	private final MemberRepository memberRepo;
	private final LoginLogRepository logRepo;
	private PasswordEncoder encoder = new PasswordEncoder();
	
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
		@Parameter(name = "userid", description= "등록된 사용자 ID"),
		@Parameter(name = "password", description= "등록된 비밀번호")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 1<br>dataList : JWTToken<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> login(
			@RequestBody loginDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		System.out.println("req : " + req);
		Optional<Member> opt =  memberRepo.findByUserId(req.userId);
		if(opt.isEmpty()) {
			res.setSuccess(false);
			res.setErrorMsg("회원 정보가 존재하지 않습니다. ID와 비밀번호를 확인해주세요.");
			return ResponseEntity.ok().body(res);
		}
		Member member = opt.get();
		if(!encoder.matches(req.getPassword(), member.getPassword())) {
			res.setSuccess(false);
			res.setErrorMsg("회원 정보가 존재하지 않습니다. ID와 비밀번호를 확인해주세요.");
			return ResponseEntity.ok().body(res);
		}
		
		// 로그인 시간 갱신
		LocalDateTime now = LocalDateTime.now();
		member.setLastLoginTime(now);
		memberRepo.save(member);
		
		// 로그 추가
		logRepo.save(LoginLog.builder()
				.member(member)
				.loginTime(now)
				.build());
		
		// 토큰 생성
		String token = JWTUtil.getJWT(member);
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
		
		// 토큰 추출 및 검증
		String token = request.getHeader("Authorization");
		if (token == null || !token.startsWith("Bearer ")) {
			res.setSuccess(false);
			res.setErrorMsg("유효한 토큰이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		// JWT에서 userid 추출
		try {
			String userid = JWTUtil.getClaim(token, JWTUtil.useridClaim);
			System.out.println("[MemberController] logout request for user: " + userid);
			
			// TokenBlacklistManager에서 토큰 무효화
			// (자동으로 TokenBlacklistManager의 invalidateToken이 호출됨)
			
			res.setSuccess(true);
			res.setErrorMsg(null);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg("로그아웃 처리 중 오류가 발생했습니다.");
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
		Member member = JWTUtil.parseToken(request, memberRepo);
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
		List<Member> list = memberRepo.findAll();
		for(Member mem : list)
			res.addData(mem);
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
		@Parameter(name = "userid", description = "등록할 사용자 ID"),
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
		Member member = JWTUtil.parseToken(request, memberRepo);
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
		if(memberRepo.findByUserId(req.userId).isPresent()) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 ID 입니다.");
			return ResponseEntity.ok().body(res);
		}
		
		memberRepo.save(Member.builder()
							.userId(req.userId)
							.password(encoder.encode(req.password))
							.userName(req.userName)
							.role(Role.ROLE_MEMBER)
							.build());
		
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
	@Operation(summary="맴버 정보 변경", description = "user_no를 이용해서 userid,password,role을 변경")
	@Parameters( {
		@Parameter(name = "user_no", description= "변경할 사용자 고유번호"),
		@Parameter(name = "userid", description= "변경할 사용자 ID"),
		@Parameter(name = "password", description= "변경할 비밀번호"),
		@Parameter(name = "userName", description = "변경할 유저명"),
		@Parameter(name = "role", description = "변경할 이용자 권한", example = "ROLE_MEMBER")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> modifyMember(
			HttpServletRequest request,
			@RequestBody modifyDTO req
			) {
		System.out.println("req : " + req);
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
		Member member = JWTUtil.parseToken(request, memberRepo);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		Optional<Member> opt = memberRepo.findById(req.userNo);
		if(opt.isEmpty()) {
			res.setSuccess(false);
			res.setErrorMsg("존재하지 않는 회원정보입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN && member.getUserNo() != req.userNo) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		if(memberRepo.findByUserId(req.userId).isPresent()) {
			res.setSuccess(false);
			res.setErrorMsg("이미 사용중인 ID 입니다.");
			return ResponseEntity.ok().body(res);
		}
		Member modifyMember = opt.get();
		modifyMember.setUserId(req.userId);
		modifyMember.setPassword(encoder.encode(req.password));
		modifyMember.setUserName(req.userName);
		modifyMember.setRole(req.role);
		memberRepo.save(modifyMember);
		
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
	@Operation(summary="맴버 정보 삭제", description = "user_no/user_id를 이용해서 회원정보를 삭제")
	@Parameters( {
		@Parameter(name = "user_no", description= "변경할 사용자 고유번호"),
		@Parameter(name = "userid", description= "변경할 사용자 ID")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
	public ResponseEntity<Object> deleteMember(
			HttpServletRequest request,
			@RequestBody deleteDTO req
			) {
		System.out.println("req : " + req);
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
		Member member = JWTUtil.parseToken(request, memberRepo);
		if(member == null){
			res.setSuccess(false);
			res.setErrorMsg("로그인이 필요합니다.");
			return ResponseEntity.ok().body(res);
		}
		Optional<Member> opt = memberRepo.findById(req.userNo);
		if(opt.isEmpty()) {
			res.setSuccess(false);
			res.setErrorMsg("존재하지 않는 회원정보입니다.");
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() != Role.ROLE_ADMIN && member.getUserNo() != req.userNo) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		Member deleteMember = opt.get();
		memberRepo.delete(deleteMember);
		
		return ResponseEntity.ok().body(res);
	}
}
