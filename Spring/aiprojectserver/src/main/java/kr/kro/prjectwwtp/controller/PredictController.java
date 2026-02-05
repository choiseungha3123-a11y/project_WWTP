package kr.kro.prjectwwtp.controller;

import java.util.TimeZone;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
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
import kr.kro.prjectwwtp.domain.responseDTO;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/predict")
@RequiredArgsConstructor
@Tag(name="PredictController", description = "예측값 조회 API")
public class PredictController {
	
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
	
	@GetMapping("/predict")
	@Operation(summary="요약", description = "설명")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = String.class))
	@ApiResponse(description = "결과 설명", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> getPredict() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();

		return ResponseEntity.ok().body(res);
	}

}
