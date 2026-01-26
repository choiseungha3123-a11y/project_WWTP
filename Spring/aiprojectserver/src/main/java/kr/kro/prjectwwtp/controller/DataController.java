package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.DataRepository;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequiredArgsConstructor
public class DataController {
	private final DataRepository dataRepo;
	
	@ExceptionHandler(MissingServletRequestParameterException.class)
	public ResponseEntity<Object> handleMissingParams(MissingServletRequestParameterException ex) {
		responseDTO res = responseDTO.builder()
				.bSuccess(false)
				.errorMsg(ex.getParameterName() + " 파라메터가 누락되었습니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@ExceptionHandler(MethodArgumentTypeMismatchException.class)
	public ResponseEntity<Object> handleMismatchParams(MethodArgumentTypeMismatchException ex) {
		responseDTO res = responseDTO.builder()
				.bSuccess(false)
				.errorMsg(ex.getName() + " 파라메터의 형식이 올바르지 않습니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@ExceptionHandler(HttpRequestMethodNotSupportedException.class)
	public ResponseEntity<Object> handleMethodNotSupported(HttpRequestMethodNotSupportedException ext) {
		responseDTO res = responseDTO.builder()
				.bSuccess(false)
				.errorMsg(" 허용되지 않는 Method 입니다.")
				.build();
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/api/data")
	public ResponseEntity<Object> getTest(
			@RequestParam String tm1,
			@RequestParam String tm2) {
		responseDTO res = responseDTO.builder()
				.bSuccess(true)
				.errorMsg(null)
				.build();
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");
		LocalDateTime start = LocalDateTime.parse(tm1, formatter);
		LocalDateTime end = LocalDateTime.parse(tm2, formatter);
		System.out.println("start : " + start);
		System.out.println("end : " + end);
		List<TmsData> list = dataRepo.findByTimeBetweenOrderByDataNoDesc(start, end);
		for(TmsData data : list)
			res.addData(data);
		return ResponseEntity.ok().body(res);
	}

}
