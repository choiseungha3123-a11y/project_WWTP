package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
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
import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/weather")
@RequiredArgsConstructor
@Tag(name="WeatherController", description = "날씨 데이터 조회용 API")
public class WeathereController {
	private final WeatherRepository weatherRepo;
	
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
	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
	@Parameters( {
		@Parameter(name = "tm1", description= "조회시작날짜(yyyyMMddHHmm)", example = "202401010000"),
		@Parameter(name = "tm2", description= "조회종료날짜(yyyyMMddHHmm)", example = "202401012359")
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : dataList에 들어 있는 값들의 개수<br>dataList : 결과값배열<br>errorMsg : success가 false 일때의 오류원인 ", content = @Content(schema = @Schema(implementation = TmsData.class)))
	public ResponseEntity<Object> getWeatherList(
			@RequestParam String tm1,
			@RequestParam String tm2) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");
		LocalDateTime start = LocalDateTime.parse(tm1, formatter);
		LocalDateTime end = LocalDateTime.parse(tm2, formatter);
		System.out.println("start : " + start);
		System.out.println("end : " + end);
		List<TmsData> list = weatherRepo.findByTimeBetweenOrderByDataNoDesc(start, end);
		for(TmsData data : list)
			res.addData(data);
		return ResponseEntity.ok().body(res);
	}

}
