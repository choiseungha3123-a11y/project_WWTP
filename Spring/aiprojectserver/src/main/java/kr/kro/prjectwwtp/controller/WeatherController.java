package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.TimeZone;

import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
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
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.Weather;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@RestController
@RestControllerAdvice
@RequestMapping("/api/weather")
@RequiredArgsConstructor
@Tag(name="WeatherController", description = "날씨 데이터 조회용 API")
public class WeatherController {
	private final WeatherService weatherService;
	
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
	@NoArgsConstructor
	@AllArgsConstructor
	static public class weatherDTO {
		@Schema(description = "고유번호", example = "1~")
		long dataNo;
		@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
		String time;
		@Schema(description = "1분 평균 기온 (C)")
		double ta;
		@Schema(description = "15분 누적 강수량 (mm)")
		double rn15m;
		@Schema(description = "60분 누적 강수량 (mm)")
		double rn60m;
		@Schema(description = "12시간 누적 강수량 (mm)")
		double rn12h;
		@Schema(description = "일 누적 강수량 (mm)")
		double rnday;
		@Schema(description = "1분 평균 상대습도 (%)")
		double hm;
		@Schema(description = "이슬점온도 (C)")
		double td; 
		
		public weatherDTO(Weather data) {
			this.dataNo = data.getDataNo();
			this.time = data.getLogTime().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
			this.ta = data.getTa();
			this.rn15m = data.getRn15m();
			this.rn60m = data.getRn60m();
			this.rn12h = data.getRn12h();
			this.rnday = data.getRnday();
			this.hm = data.getHm();
			this.td = data.getTd();
		}
	}
	
	@GetMapping("/list")
	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
	@Parameter(name = "tm1", description= "조회시작날짜(yyyyMMddHHmm)", example = "202401010000")
	@Parameter(name = "tm2", description= "조회종료날짜(yyyyMMddHHmm)", example = "202401012359")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
		@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = weatherDTO.class)))
	})
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
		//List<Weather> list = weatherRepo.findByTimeBetweenOrderByDataNoDesc(start, end);
		List<Weather> list = weatherService.findByLogTimeBetween(start, end);
		for(Weather data : list)
		{
			weatherDTO d = new weatherDTO(data);
			res.addData(d);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@PatchMapping("/modify")
	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
	@Parameter(name = "Authorization", description= "{jwtToken}", example = "Bearer ey~~~")
	@Parameter(name = "Content-Type", description= "application/json", schema = @Schema(implementation = weatherDTO.class))
	@ApiResponse(description = "success, errorMsg 값만 체크", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> modifyWeatherData(
			HttpServletRequest request,
			@RequestBody weatherDTO req) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(req.dataNo == 0) {
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
		if(member.getRole() != Role.ROLE_ADMIN) {
			res.setSuccess(false);
			res.setErrorMsg("권한이 없습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		Weather data = weatherService.findById(req.dataNo);
		if(data == null) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		try {
			weatherService.modifyWeahter(data, req.ta, req.rn15m, req.rn60m, req.rn12h, req.rnday, req.hm, req.td);
		} catch(Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		
		return ResponseEntity.ok().body(res);
	}
	

}
