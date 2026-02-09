package kr.kro.prjectwwtp.controller;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.RestClient;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import com.fasterxml.jackson.databind.ObjectMapper;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.controller.WeatherController.WeatherDTO;
import kr.kro.prjectwwtp.domain.FlowImputate;
import kr.kro.prjectwwtp.domain.FlowLog;
import kr.kro.prjectwwtp.domain.FlowOrigin;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.fastApiResponseDTO;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.FlowLogRepository;
import kr.kro.prjectwwtp.service.FastApiService;
import kr.kro.prjectwwtp.service.FlowOriginService;
import kr.kro.prjectwwtp.service.FlowSummaryService;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import reactor.netty.http.client.HttpClient;

@RestController
@RestControllerAdvice
@RequestMapping("/api/flowOrigin")
@RequiredArgsConstructor
@Tag(name="FlowOriginController", description = "유량 수치 처리 API")
public class FlowOriginController {
	private final FlowOriginService flowOriginService;
	private final FlowLogRepository logRepository;
	private final FlowSummaryService flowSummaryService;
	private final WeatherService weaterhService;
	private final FastApiService apiService;
	
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

	@PostMapping("/upload")
	@Operation(summary="실제 측정 데이터 upload", description = ".csv 파일을 업로드하여 실제 측정 데이터를 저장합니다.")
	@Parameter(name = "file", description= ".csv 파일명", schema = @Schema(implementation = MultipartFile.class))
	@ApiResponse(description = "dataList[0]에 saveCount : XXXX 로 저장된 수를 전달", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class)))
	public ResponseEntity<Object> postFlowOriginUpload(
			HttpServletRequest request,
			MultipartFile file) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
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
		try {
			int saveCount = flowOriginService.saveFromCsv(file);
			logRepository.save(FlowLog.builder()
									.type("upload")
									.member(member)
									.count(saveCount)
									.build());
			res.addData("saveCount : " + saveCount);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/list")
	@Operation(summary="실제 측정 데이터 조회", description = "저장된 실제 측정 데이터를 조회합니다.")
	@Parameter(name = "time", description= "조회날짜(yyyyMMdd)", example = "20240101")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
		@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = FlowOrigin.class)))
	})
	public ResponseEntity<Object> getFlowOriginList(
			HttpServletRequest request,
			@RequestParam String time) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
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
		try {
			List<FlowOrigin> list = flowOriginService.getFlowOriginListByDate(time);
			for(FlowOrigin t : list) {
				res.addData(t);
			}
			logRepository.save(FlowLog.builder()
					.type("list")
					.member(member)
					.time(time)
					.count(list.size())
					.build());
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/flowList")
	@Operation(summary="어제부터의 실시간 정보와 내일까지의 예상 정보를 요청", description = "결측/이상 값을 처리한 데이터를 조회합니다. 데이터가 없으면 보간을 수행합니다.")
/*
	public ResponseEntity<Object> getFlowList(
			HttpServletRequest request) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
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
		
		try {
			LocalDateTime fakeNow = flowSummaryService.getFakeNow();
			LocalDateTime now = LocalDateTime.now();
			fakeNow = fakeNow.withHour(now.getHour());
			fakeNow = fakeNow.withMinute(now.getMinute());
			
			List<FlowImputate> list = flowOriginService.getFlowImputateListByDate(fakeNow);
						
//			String csvFilePath = "Downloads/imputated_data_" + time + ".csv";
//			flowOriginService.saveToCsv(list, csvFilePath);
			
			for(FlowImputate t : list) {
				res.addData(t);
			}
			logRepository.save(FlowLog.builder()
					.type("imputate")
					.member(member)
					.time(fakeNow.toString())
					.count(list.size())
					.build());
								
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}		 		
		return ResponseEntity.ok().body(res);
	}
*/	
	
	public ResponseEntity<Object> getFlowList() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		try {
			LocalDateTime now = LocalDateTime.now();
			LocalDateTime fakeNow = flowSummaryService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute());
			List<FlowImputate> flowList = flowOriginService.getFlowImputateListByDate(fakeNow);
			List<WeatherDTO> aws368 = weaterhService.findWeatherDTOByStnAndLogTimeBetween(368, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			List<WeatherDTO> aws541 = weaterhService.findWeatherDTOByStnAndLogTimeBetween(541, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			List<WeatherDTO> aws569 = weaterhService.findWeatherDTOByStnAndLogTimeBetween(569, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			requestFlow(aws368, aws541, aws569, flowList);
			
//			res.addData(flowList);
//			logRepository.save(FlowLog.builder()
//					.type("imputate")
//					.time(fakeNow.toString())
//					.count(flowList.size())
//					.build());
								
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}		 	
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@ToString
	static public class predictIn {
		private String predictIn;
		private Input in;
		public predictIn(Input in) {
			this.in = in;
		}
	}
	
	@Getter
	@Setter
	@ToString
	static public class awsListByStd {
		public List<WeatherDTO> STN_368;
		public List<WeatherDTO> STN_541;
		public List<WeatherDTO> STN_569;
		public awsListByStd(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569) {
			this.STN_368 = aws368;
			this.STN_541 = aws541;
			this.STN_569 = aws569;
		}
	}
	
	@Getter
	@Setter
	@ToString
	static public class Input {
		public awsListByStd awsList;
		public List<FlowImputate> dataList;
		public Input(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<FlowImputate> flowList) {
			this.awsList = new awsListByStd( aws368, aws541, aws569);
			this.dataList = flowList;
		}
	}
	
	
	public void requestFlow(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<FlowImputate>flowList) {
		Input input = new Input(aws368, aws541, aws569, flowList);
		predictIn pIn = new predictIn(input);
		
		fastApiResponseDTO response = apiService.getPrectFlow(pIn);
		System.out.println(response);
	}

}