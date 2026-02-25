//package kr.kro.prjectwwtp.controller;
//
//import java.time.LocalDateTime;
//import java.time.format.DateTimeFormatter;
//import java.util.List;
//
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.HttpRequestMethodNotSupportedException;
//import org.springframework.web.bind.MissingServletRequestParameterException;
//import org.springframework.web.bind.annotation.ExceptionHandler;
//import org.springframework.web.bind.annotation.GetMapping;
//import org.springframework.web.bind.annotation.PatchMapping;
//import org.springframework.web.bind.annotation.RequestBody;
//import org.springframework.web.bind.annotation.RequestMapping;
//import org.springframework.web.bind.annotation.RequestParam;
//import org.springframework.web.bind.annotation.RestController;
//import org.springframework.web.bind.annotation.RestControllerAdvice;
//import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
//
//import com.fasterxml.jackson.annotation.JsonIgnore;
//import com.fasterxml.jackson.annotation.JsonProperty;
//
//import io.swagger.v3.oas.annotations.Operation;
//import io.swagger.v3.oas.annotations.Parameter;
//import io.swagger.v3.oas.annotations.media.Content;
//import io.swagger.v3.oas.annotations.media.Schema;
//import io.swagger.v3.oas.annotations.responses.ApiResponse;
//import io.swagger.v3.oas.annotations.responses.ApiResponses;
//import io.swagger.v3.oas.annotations.tags.Tag;
//import jakarta.servlet.http.HttpServletRequest;
//import kr.kro.prjectwwtp.domain.Member;
//import kr.kro.prjectwwtp.domain.Role;
//import kr.kro.prjectwwtp.domain.Weather;
//import kr.kro.prjectwwtp.domain.responseDTO;
//import kr.kro.prjectwwtp.service.LogService;
//import kr.kro.prjectwwtp.service.WeatherService;
//import kr.kro.prjectwwtp.util.JWTUtil;
//import lombok.AllArgsConstructor;
//import lombok.Getter;
//import lombok.NoArgsConstructor;
//import lombok.RequiredArgsConstructor;
//import lombok.Setter;
//import lombok.ToString;
//
//@RestController
//@RestControllerAdvice
//@RequestMapping("/api/weather")
//@RequiredArgsConstructor
//@Tag(name="WeatherController", description = "날씨 데이터 조회용 API")
//public class WeatherController {
//	private final WeatherService weatherService;
//	private final LogService logService;
//	
//	@ExceptionHandler(MissingServletRequestParameterException.class)
//	public ResponseEntity<Object> handleMissingParams(MissingServletRequestParameterException ex) {
//		responseDTO res = responseDTO.builder()
//				.success(false)
//				.errorMsg(ex.getParameterName() + " 파라메터가 누락되었습니다.")
//				.build();
//		return ResponseEntity.ok().body(res);
//	}
//	
//	@ExceptionHandler(MethodArgumentTypeMismatchException.class)
//	public ResponseEntity<Object> handleMismatchParams(MethodArgumentTypeMismatchException ex) {
//		responseDTO res = responseDTO.builder()
//				.success(false)
//				.errorMsg(ex.getName() + " 파라메터의 형식이 올바르지 않습니다.")
//				.build();
//		return ResponseEntity.ok().body(res);
//	}
//	
//	@ExceptionHandler(HttpRequestMethodNotSupportedException.class)
//	public ResponseEntity<Object> handleMethodNotSupported(HttpRequestMethodNotSupportedException ext) {
//		responseDTO res = responseDTO.builder()
//				.success(false)
//				.errorMsg(" 허용되지 않는 Method 입니다.")
//				.build();
//		return ResponseEntity.ok().body(res);
//	}
//	
//	
//	@GetMapping("/list")
//	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
//	@Parameter(name = "tm1", description= "조회시작날짜(yyyyMMddHHmm)", example = "202401010000")
//	@Parameter(name = "tm2", description= "조회종료날짜(yyyyMMddHHmm)", example = "202401012359")
//	@ApiResponses({
//		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
//		@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = WeatherDTO.class)))
//	})
//	public ResponseEntity<Object> getWeatherList(
//			@RequestParam String tm1,
//			@RequestParam String tm2) {
//		responseDTO res = responseDTO.builder()
//				.success(true)
//				.errorMsg(null)
//				.build();
//		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");
//		LocalDateTime start = LocalDateTime.parse(tm1, formatter);
//		LocalDateTime end = LocalDateTime.parse(tm2, formatter);
//		System.out.println("start : " + start);
//		System.out.println("end : " + end);
//		List<WeatherDTO> list = weatherService.findWeatherDTOByLogTimeBetween(start, end);
//		for(WeatherDTO data : list)
//		{
//			res.addData(data);
//		}
//		return ResponseEntity.ok().body(res);
//	}
//	
//	@PatchMapping("/modify")
//	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
//	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
//	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = WeatherDTO.class))
//	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
//	public ResponseEntity<Object> modifyWeatherData(
//			HttpServletRequest request,
//			@RequestBody WeatherDTO req) {
//		responseDTO res = responseDTO.builder()
//				.success(true)
//				.errorMsg(null)
//				.build();
//		if(req.dataNo == 0) {
//			res.setSuccess(false);
//			res.setErrorMsg("정보가 올바르지 않습니다.");
//			return ResponseEntity.ok().body(res);
//		}
//		if(JWTUtil.isExpired(request))
//		{
//			res.setSuccess(false);
//			res.setErrorMsg("토큰이 만료되었습니다.");
//			return ResponseEntity.ok().body(res);
//		}
//		Member member = JWTUtil.parseToken(request);
//		if(member == null){
//			res.setSuccess(false);
//			res.setErrorMsg("로그인이 필요합니다.");
//			return ResponseEntity.ok().body(res);
//		}
//		if(member.getRole() != Role.ROLE_ADMIN) {
//			res.setSuccess(false);
//			res.setErrorMsg("권한이 없습니다.");
//			return ResponseEntity.ok().body(res);
//		}
//		
//		Weather data = weatherService.findById(req.dataNo);
//		if(data == null) {
//			res.setSuccess(false);
//			res.setErrorMsg("정보가 올바르지 않습니다.");
//			return ResponseEntity.ok().body(res);
//		}
//		
//		try {
//			weatherService.modifyWeahter(data, req.ta, req.rn15m, req.rn60m, req.rn12h, req.rnday, req.hm, req.td);
//		} catch(Exception e) {
//			res.setSuccess(false);
//			res.setErrorMsg(e.getMessage());
//			logService.addErrorLog("WeatherController.java", "modifyWeatherData()", e.getMessage());
//		}
//		
//		return ResponseEntity.ok().body(res);
//	}
//	
//
//}
