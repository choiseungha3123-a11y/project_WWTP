package kr.kro.prjectwwtp.controller;

import java.util.TimeZone;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
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
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;
import kr.kro.prjectwwtp.domain.PageDTO;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.MemoService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@RestController
@RestControllerAdvice
@RequestMapping("/api/memo")
@RequiredArgsConstructor
@Tag(name="MemoController", description = "회원간 메모 관리 API")
public class MemoController {
	private final MemoService memoService;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}
	
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
	
	@GetMapping("/list")
	@Operation(summary="메모 데이터 조회", description = "다른 이용자들에게 보여줄 메모 데이터를 조회합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "page", description= "조회할 페이지수", example = "0")
	@Parameter(name = "count", description= "페이지 별로 보여줄 메모의 수", example = "10")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
		@ApiResponse(responseCode = "201", description = "dataList[0]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = PageDTO.class))),
		@ApiResponse(responseCode = "202", description = "dataList[0].items[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = Memo.class)))
	})
	public ResponseEntity<Object> getMemoList(
			HttpServletRequest request,
			@RequestParam int page,
			@RequestParam int count) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		System.out.println("token : " + request.getHeader("Authorization"));
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		Pageable pageable = PageRequest.of(page, count);

		PageDTO<Memo> pageList = memoService.findByDisableMemberIsNull(member, pageable);
		res.addData(pageList);
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memoCreateDTO {
		@Schema(name = "content", description = "메모 내용", example = "신규 메모")
		private String content;
	}
	
	@PutMapping("/create")
	@Operation(summary="메모 작성", description = "새로운 메모를 작성합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoCreateDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> putMemoCreate(
			HttpServletRequest request,
			@RequestBody memoCreateDTO req) {
		System.out.println("token : " + request.getHeader("Authorization"));
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		try {
			memoService.addMemo(member, req.content);	
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memoModifyDTO {
		@Schema(name = "memoNo", description = "메모 고유번호", example = "1~")
		private long memoNo;
		@Schema(name = "content", description = "수정 메모 내용", example = "수정 메모")
		private String content;
	}
	
	@PostMapping("/modify")
	@Operation(summary="메모 수정", description = "작성된 메모를 수정합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoModifyDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postMemoModify(
			HttpServletRequest request,
			@RequestBody memoModifyDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		try {
			memoService.modifyMemo(member, req.memoNo, req.content);
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memoDisableDTO {
		@Schema(name = "memoNo", description = "메모 고유번호", example = "1~")
		private long memoNo;
	}
	
	@PostMapping("/disable")
	@Operation(summary="메모 비활성화", description = "작성된 메모를 비활성화합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoDisableDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postMemoDisable(
			HttpServletRequest request,
			@RequestBody memoDisableDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		try {
			memoService.disableMemo(member, req.memoNo);
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	@PostMapping("/delete")
	@Operation(summary="메모 삭제", description = "작성된 메모를 삭제합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoDisableDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postMemoDelete(
			HttpServletRequest request,
			@RequestBody memoDisableDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		try {
			memoService.deleteMemo(member, req.memoNo);
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/oldList")
	@Operation(summary="비활성화된 메모 조회", description = "비활성화된 메모 데이터를 조회합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "page", description= "조회할 페이지수", example = "0")
	@Parameter(name = "count", description= "페이지 별로 보여줄 메모의 수", example = "10")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
		@ApiResponse(responseCode = "201", description = "dataList[0]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = PageDTO.class))),
		@ApiResponse(responseCode = "202", description = "dataList[0].items[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = Memo.class)))
	})
	public ResponseEntity<Object> getMemoOldList(
			HttpServletRequest request,
			@RequestParam int page,
			@RequestParam int count) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		// 토큰 추출 및 검증
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
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		Pageable pageable = PageRequest.of(page, count);

		PageDTO<Memo> pageList = memoService.findByDisableMemberIsNotNull(member, pageable);
		res.addData(pageList);
		
		return ResponseEntity.ok().body(res);
	}
}
