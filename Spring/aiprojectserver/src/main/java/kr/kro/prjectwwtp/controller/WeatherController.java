package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;

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
import io.swagger.v3.oas.annotations.Parameters;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

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
	
	@Getter
	@Setter
	@ToString
	static public class weatherDTO {
		long dataNo;
		LocalDateTime time;
		double ta;
		double rn15m;
		double rn60m;
		double rn12h;
		double rnday;
		double hm;
		double td; 
		
		public weatherDTO(TmsData data) {
			this.dataNo = data.getDataNo();
			this.time = data.getTime();
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
	@Parameters( {
		@Parameter(name = "tm1", description= "조회시작날짜(yyyyMMddHHmm)", example = "202401010000"),
		@Parameter(name = "tm2", description= "조회종료날짜(yyyyMMddHHmm)", example = "202401012359")
	})
	@ApiResponse(description = "", content = @Content(schema = @Schema(implementation = responseDTO.class)))
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
		List<TmsData> list = weatherRepo.findByTimeBetween(start, end);
		System.out.println("list size : " + list.size());
		for(TmsData data : list)
		{
			weatherDTO d = new weatherDTO(data);
			System.out.println(d);
			res.addData(d);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@PatchMapping("/modify")
	@Operation(summary="날씨 데이터 조회", description = "DB에 저장된 기상청 날씨 정보 조회")
	@Parameters( {
		@Parameter(name = "dataNo", description= "고유번호(long)", example = ""),
		@Parameter(name = "ta", description= "1분 평균 기온(double)", example = ""),
		@Parameter(name = "rn15m", description= "15분 누적 강수량(double)", example = ""),
		@Parameter(name = "rn60m", description= "60분 누적 강수량(double)", example = ""),
		@Parameter(name = "rn12h", description= "12시간 누적 강수량(double)", example = ""),
		@Parameter(name = "rnday", description= "일 누적 강수량(double)", example = ""),
		@Parameter(name = "hm", description= "1분 평균 상대습도(double)", example = ""),
		@Parameter(name = "td", description= "이슬점온도(double)", example = ""),
	})
	@ApiResponse(description = "success : 성공/실패<br>dataSize : 0<br>dataList : NULL<br>errorMsg : success가 false 일때의 오류원인 ")
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
		
		Optional<TmsData> opt = weatherRepo.findById(req.dataNo);
		if(opt.isEmpty()) {
			res.setSuccess(false);
			res.setErrorMsg("정보가 올바르지 않습니다.");
			return ResponseEntity.ok().body(res);
		}
		
		TmsData data = opt.get();
		data.setTa(req.ta);
		data.setRn15m(req.rn15m);
		data.setRn60m(req.rn60m);
		data.setRn12h(req.rn12h);
		data.setRnday(req.rnday);
		data.setHm(req.hm);
		data.setTd(req.td);
		weatherRepo.save(data);
		
		return ResponseEntity.ok().body(res);
	}
	

}
