package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MultipartFile;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.FlowImputate;
import kr.kro.prjectwwtp.domain.FlowPredict;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;
import kr.kro.prjectwwtp.domain.PageDTO;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsPredict;
import kr.kro.prjectwwtp.domain.WeatherDTO;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.FlowService;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.service.MailService;
import kr.kro.prjectwwtp.service.MemberService;
import kr.kro.prjectwwtp.service.MemoService;
import kr.kro.prjectwwtp.service.TmsService;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@RestController
@RestControllerAdvice
@RequestMapping("/api/board")
@RequiredArgsConstructor
@Tag(name="DashBoardController", description = "회원간 메모 관리 API")
public class DashBoardController {
	private final MemoService memoService;
	private final TmsService tmsService;
	private final FlowService flowService;
	private final WeatherService weatherService;
	private final LogService logService;
	private final MemberService memberService;
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
	@Operation(summary="서버의 동작 상태를 체크")
	public ResponseEntity<Object> healthCheck() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/memo/list")
	@Operation(summary="메모 데이터 조회", description = "다른 이용자들에게 보여줄 메모 데이터를 조회합니다.")
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
	
	@GetMapping("memo/image")
	@Operation(summary="메모에 첨부된 사진 확인", description = "첨부 파일 확인")
	@Parameter(name = "memo_no", description = "메모 고유 번호", example = "1~")
	public ResponseEntity<Object> getMemoImage(
			@RequestParam long memo_no) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		Memo memo = memoService.findByMemoNo(memo_no);
		if(memo == null) {
			res.setSuccess(false);
			res.setErrorMsg("메모 고유번호가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		String fileName = memo.getFileName();
		String fileType = memo.getFileType();
		byte[] imageData = memo.getImageData();
		res.addData(fileName);
		res.addData(fileType);
		res.addData(imageData);
		
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class memoCreateDTO {
		@Schema(name = "content", description = "메모 내용", example = "신규 메모")
		private String content;
		private MultipartFile file;
	}
	
	@PutMapping("/memo/create")
	@Operation(summary="메모 작성", description = "새로운 메모를 작성합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoCreateDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> putMemoCreate(
			HttpServletRequest request,
			@ModelAttribute memoCreateDTO req) {
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
			memoService.addMemo(member, req.content, req.file);	
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
			logService.addErrorLog("DashBoardController.java", "putMemoCreate()", e.getMessage());
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
		private MultipartFile file;
	}
	
	@PostMapping("/memo/modify")
	@Operation(summary="메모 수정", description = "작성된 메모를 수정합니다.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = memoModifyDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postMemoModify(
			HttpServletRequest request,
			@ModelAttribute memoModifyDTO req) {
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
			memoService.modifyMemo(member, req.memoNo, req.content, req.file);
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
			logService.addErrorLog("DashBoardController.java", "postMemoModify()", e.getMessage());
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
	
	@PostMapping("/memo/disable")
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
			logService.addErrorLog("DashBoardController.java", "postMemoDisable()", e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	@PostMapping("/memo/delete")
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
			logService.addErrorLog("DashBoardController.java", "postMemoDelete()", e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/memo/oldList")
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
	
	@GetMapping("/boardView")
	@Operation(summary="대시보드에 보여질 데이터 구성", description = "24시간전부터의 측정값, 12간후의 예측값")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
	})
	public ResponseEntity<Object> getBoardView(
			HttpServletRequest request) {
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
			LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
			LocalDateTime start = now.minusDays(1).plusMinutes(1);
			LocalDateTime end = now.plusDays(1).minusMinutes(1);
			LocalDateTime fakeTmsNow = tmsService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute());
			LocalDateTime fakeFlowNow = flowService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute());
			List<TmsImputate> tmsImputateList = tmsService.getTmsImputateListByDateForDashBoard(fakeTmsNow);
			if(test.visible) {
				TmsImputate last = tmsImputateList.get(tmsImputateList.size() - 1);
				tmsImputateList.add(TmsImputate.builder()
						.tmsTime(last.getTmsTime())
						.strtime(last.getStrtime())
						.flux(last.getFlux())
						.ph(test.ph)
						.ss(test.ss)
						.toc(test.toc)
						.tn(test.tn)
						.tp(test.tp)
						.build());
			}
			List<TmsPredict> tmsPredictList = tmsService.findPredictList(start, end);
			List<FlowImputate> flowImputateList = flowService.getFlowImputateListByDateForDashBoard(fakeFlowNow);
			List<FlowPredict> flowPredictList = flowService.findPredictList(start, end);
			List<WeatherDTO> aws368 = weatherService.findWeatherDTOByStnAndLogTimeBetween(368, start, now);
			if(test.visible) {	
				aws368.add(WeatherDTO.builder()
						.rn15m(test.rain)
						.build());
			}
			
			res.addData(tmsImputateList);
			res.addData(tmsPredictList);
			res.addData(flowImputateList);
			res.addData(flowPredictList);
			res.addData(aws368);
			
		}catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
			logService.addErrorLog("DashBoardController.java", "getBoardView()", e.getMessage());
		}
		
		return ResponseEntity.ok().body(res);
	}
	

	
	@Getter
	@Setter
	@ToString
	static public class SendToDTO {
		@Schema(name = "userNo", description = "메일을 받을 사원의 고유번호", example = "1~")
		private long userNo;
	}
	
	@PostMapping("/sendReportTo")
	@Operation(summary="보고서 메일 발송", description = "현재 시점 이후 12시간의 예측 보고서를 등록한 메일로 발송.")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	public ResponseEntity<Object> postSendReportTo(
			HttpServletRequest request,
			@RequestBody SendToDTO req) {
		String errorMsg = null;
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
			errorMsg = "로그인이 필요합니다.";
			res.setErrorMsg(errorMsg);
			return ResponseEntity.ok().body(res);
		}
		if(member.getRole() == Role.ROLE_VIEWER) {
			res.setSuccess(false);
			errorMsg = "권한이 올바르지 않습니다.";
			res.setErrorMsg(errorMsg);
			return ResponseEntity.ok().body(res);
		}
		Member recvMember = memberService.findByNo(req.userNo);
		if(recvMember.getUserEmail() == null || !recvMember.isValidateEmail()) {
			res.setSuccess(false);
			errorMsg = "인증된 메일 정보가 없습니다.";
			res.setErrorMsg(errorMsg);
			return ResponseEntity.ok().body(res);
		}
		try {
			try {
				LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
				LocalDateTime end = now.plusDays(1).minusMinutes(1);
				List<TmsPredict> tmsList = tmsService.findPredictList(now, end);
				List<FlowPredict> flowList = flowService.findPredictList(now, end);

				String html = mailService.reportChart(tmsList, flowList);
				String fileName = "chart" + now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + ".html";
				
				String subject = "Report From FlowWater";
				String body = mailService.reportBody(recvMember);
				
				mailService.sendEmailWithAttachment(recvMember, subject, body, html, fileName);
			}catch(Exception e) {
				e.printStackTrace();
				logService.addErrorLog("MemberController.java", "makeReportMessage()", e.getMessage());
			}
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
			logService.addErrorLog("DashBoardController.java", "postMemoModify()", e.getMessage());
		}
		
		
		return ResponseEntity.ok().body(res);
	}
	
	
	@Getter
	@Setter
	@ToString
	public static class TestDTO {
		boolean visible = false;
		double rain = 0.0;
		double ph = 0.0;
		double ss = 0.0;
		double toc = 0.0;
		double tn = 0.0;
		double tp = 0.0;
	}
	
	public static TestDTO test = new TestDTO();
	
	@PostMapping("/testValue")
	public ResponseEntity<Object> postTestValue(
			@RequestBody TestDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		
		test = req;
		System.out.println("test : " + test);
		
		return ResponseEntity.ok().body(res);
	}
	
}
