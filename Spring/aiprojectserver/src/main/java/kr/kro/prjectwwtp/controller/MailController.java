package kr.kro.prjectwwtp.controller;

import java.util.TimeZone;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.MailService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@RestController
@RestControllerAdvice
@RequestMapping("/api/mail")
@RequiredArgsConstructor
@Tag(name="MainController", description = "메일 전송 API")
public class MailController {
	private final MailService mailService;
	
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
	
	
	@Getter
	@Setter
	@ToString
	static public class sendToMailDTO {
		@Schema(name = "mail", description = "보낼 메일 주소", example = "xxx@xxx.xxx")
		private String mail;
		@Schema(name = "subject", description = "보낼 메일 제목.", example = "경고!!")
		private String subject;
		@Schema(name = "body", description = "보낼 메일 내용.", example = "경고!! 3시간 내에 수위가 위험할수 있습니다.")
		private String body;
	}
	
	
	@GetMapping("/sendTo")
	@Operation(summary="메일 보내기", description = "경고 상황을 알려주는 메일을 보냅니다.")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = sendToMailDTO.class))
	@ApiResponse(description = "결과 설명", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> getPredict(
			HttpServletRequest request,
			@RequestBody sendToMailDTO req) {
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
			mailService.sendEmail(req.getMail(), req.getSubject(), req.getBody());
		}
		catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}

		return ResponseEntity.ok().body(res);
	}

}
