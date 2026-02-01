package kr.kro.prjectwwtp.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MultipartFile;

import io.swagger.v3.oas.annotations.tags.Tag;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.TmsImportResult;
import kr.kro.prjectwwtp.service.TmsOriginService;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/tmsOrigin")
@RequiredArgsConstructor
@Tag(name="TmsOriginController", description = "TMS Origin API")
public class TmsOriginController {
	private final TmsOriginService tmsOriginService;
	
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

	@PostMapping("/upload")
	public ResponseEntity<Object> uploadCsv(MultipartFile file) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		try {
			TmsImportResult stat = tmsOriginService.saveFromCsv(file);
			res.setDataSize(stat.getSavedCount());
			res.addData(stat);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}

}