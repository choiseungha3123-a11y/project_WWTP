package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

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
import kr.kro.prjectwwtp.domain.Input;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.domain.TmsPredict;
import kr.kro.prjectwwtp.domain.fastApiResponseDTO;
import kr.kro.prjectwwtp.domain.predictIn;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.FastApiService;
import kr.kro.prjectwwtp.service.FlowService;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.service.TmsService;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/tmsOrigin")
@RequiredArgsConstructor
@Tag(name="TmsOriginController", description = "TMS 수치 처리 API")
public class TmsController {
	private final LogService logService;
	private final TmsService tmsService;
	private final FlowService flowService;
	private final WeatherService weatherService;
	private final FastApiService apiService;
	
	@Value("${predict.enable}")
	private boolean enable;
	
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
	public ResponseEntity<Object> postTmsOriginUpload(
			HttpServletRequest request,
			MultipartFile file) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		Member member = null;
		int saveCount = 0;
		String errorMsg = null;
		try {
			if(JWTUtil.isExpired(request))
			{
				res.setSuccess(false);
				errorMsg = "토큰이 만료되었습니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			member = JWTUtil.parseToken(request);
			if(member == null){
				res.setSuccess(false);
				res.setSuccess(false);
				errorMsg = "로그인이 필요합니다.";
				return ResponseEntity.ok().body(res);
			}
			if(member.getRole() != Role.ROLE_ADMIN) {
				res.setSuccess(false);
				errorMsg = "권한이 없습니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			saveCount = tmsService.saveFromCsv(file);
			res.addData("saveCount : " + saveCount);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		} finally {
			logService.addTmsLog(member, "upload", saveCount, errorMsg);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/list")
	@Operation(summary="실제 측정 데이터 조회", description = "저장된 실제 측정 데이터를 조회합니다.")
	@Parameter(name = "time", description= "조회날짜(yyyyMMdd)", example = "20240101")
	@ApiResponses({
		@ApiResponse(responseCode = "200", description = "결과", content = @Content(mediaType = "application/json", schema = @Schema(implementation = responseDTO.class))),
		@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = TmsOrigin.class)))
	})
	public ResponseEntity<Object> getTmsOriginList(
			HttpServletRequest request,
			@RequestParam String time) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		Member member = null;
		int listSize = 0;
		String errorMsg = null;
		try {
			if(JWTUtil.isExpired(request))
			{
				res.setSuccess(false);
				errorMsg = "토큰이 만료되었습니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			member = JWTUtil.parseToken(request);
			if(member == null){
				res.setSuccess(false);
				errorMsg = "로그인이 필요합니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			if(member.getRole() != Role.ROLE_ADMIN) {
				res.setSuccess(false);
				errorMsg = "권한이 없습니다.";
				res.setErrorMsg(errorMsg);
				return ResponseEntity.ok().body(res);
			}
			List<TmsOrigin> list = tmsService.getTmsOriginListByDate(time);
			for(TmsOrigin t : list) {
				res.addData(t);
			}
			listSize = list.size();
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		} finally {
			logService.addTmsLog(member, "list", listSize, errorMsg);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@Scheduled(cron = "${scheduler.fakeday.cron}")
	public void makeFakeDate() {
		System.out.println("makeFakeDate");
		LocalDateTime now = LocalDateTime.now();
		LocalDateTime fakeTmeNow = tmsService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute());
		System.out.println("fakeTmeNow : " + fakeTmeNow);
		
		// 조회할 날짜(fakeTmeNow를 기준으로 이전 날짜와 해당 날짜의 보간 데이터 구성
		if(!tmsService.existsByTmsTime(fakeTmeNow)) {
			List<TmsImputate> list = tmsService.imputate(fakeTmeNow);
			tmsService.saveTmsImputateList(list);
		}
		if(!tmsService.existsByTmsTime(fakeTmeNow.minusDays(1))) {
			List<TmsImputate> list = tmsService.imputate(fakeTmeNow.minusDays(1));
			tmsService.saveTmsImputateList(list);
		}
		
		LocalDateTime fakeFlowNow = flowService.getFakeNow()
				.withHour(now.getHour())
				.withMinute(now.getMinute());
		System.out.println("fakeFlowNow : " + fakeFlowNow);
				
		// 조회할 날짜(fakeTmeNow를 기준으로 이전 날짜와 해당 날짜의 보간 데이터 구성
		if(!flowService.existsByFlowTime(fakeFlowNow)) {
			List<FlowImputate> list = flowService.imputate(fakeFlowNow);
			flowService.saveFlowImputateList(list);
		}
		if(!flowService.existsByFlowTime(fakeFlowNow.minusDays(1))) {
			List<FlowImputate> list = flowService.imputate(fakeFlowNow.minusDays(1));
			flowService.saveFlowImputateList(list);
		}
	}
	
	@Scheduled(cron = "${scheduler.predict.cron}")
	public void getTmsPredict() {
		if(!enable) return;
		try {
			LocalDateTime now = LocalDateTime.now();
			LocalDateTime fakeNow = tmsService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute());
			List<TmsImputate> tmsList = tmsService.getTmsImputateListByDate(fakeNow);
			List<WeatherDTO> aws368 = weatherService.findWeatherDTOByStnAndLogTimeBetween(368, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			List<WeatherDTO> aws541 = weatherService.findWeatherDTOByStnAndLogTimeBetween(541, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			List<WeatherDTO> aws569 = weatherService.findWeatherDTOByStnAndLogTimeBetween(569, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			
			requestTms(aws368, aws541, aws569, tmsList);
								
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@GetMapping("/tmsList")
	@Operation(summary="어제부터의 실시간 정보와 내일까지의 예상 정보를 요청", description = "결측/이상 값을 처리한 데이터를 조회합니다. 데이터가 없으면 보간을 수행합니다.")
	public ResponseEntity<Object> getTmsList() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
		LocalDateTime end = now.plusDays(1).minusMinutes(1);
		List<TmsPredict> list = tmsService.findPredictList(now, end);
		res.addData(list);
		return ResponseEntity.ok().body(res);
	}
	
	public void requestTms(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<TmsImputate>tmsList) {
		String errorMsg = null;
		int predictSize = 0;
		try {
		Input<TmsImputate> input = new Input<>(aws368, aws541, aws569, tmsList);
			predictIn<TmsImputate> pIn = new predictIn<>(input);
			
			fastApiResponseDTO response = apiService.getPredict("/predict/tms", pIn);
			if(response.isOk()) {
				TmsPredict[] predictions = extractPredictions(response);
				predictSize = predictions.length;
				System.out.println("예측값 (0.5h~12.0h): " + java.util.Arrays.toString(predictions));
				tmsService.savePredictList(predictions);
			}
		}catch(Exception e) {
			errorMsg = e.getMessage();
		}
		finally {
			logService.addTmsLog(null, "predict", predictSize, errorMsg);
		}
	}
	
	/**
	 * FastAPI 응답에서 predictions 값을 1h~12h 순으로 double 배열로 추출
	 * @param response FastAPI 응답 DTO
	 * @return predictions 배열 (크기: 12), 추출 실패 시 null
	 */
	private TmsPredict[] extractPredictions(fastApiResponseDTO response) {
		int predictSize = 24;
		boolean checkOutLier = false;
		TmsPredict[] predictions = new TmsPredict[predictSize];
		
		ObjectMapper mapper = new ObjectMapper();
		
		try {
			if(response == null || response.getOutput() == null) {
				System.err.println("응답 또는 output이 null입니다");
				return null;
			}
			
			Map<String, Object> mapOutput = response.getOutput();
			Map<String, Object> mapPredictions = mapper.convertValue(mapOutput.get("predictions"),new TypeReference<>() {});
			Map<String, Object> mapToc = mapper.convertValue(mapPredictions.get("toc"),new TypeReference<>() {});
			Map<String, Object> mapSs = mapper.convertValue(mapPredictions.get("ss"),new TypeReference<>() {});
			Map<String, Object> mapTn = mapper.convertValue(mapPredictions.get("tn"),new TypeReference<>() {});
			Map<String, Object> mapTp = mapper.convertValue(mapPredictions.get("tp"),new TypeReference<>() {});
			Map<String, Object> mapFlux = mapper.convertValue(mapPredictions.get("flux"),new TypeReference<>() {});
			Map<String, Object> mapPh = mapper.convertValue(mapPredictions.get("ph"),new TypeReference<>() {});
			LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
			
			// output에서 predictions 데이터 추출
			for(int index = 1; index <= predictSize; index++) {
				String key = index/2 + (index % 2 == 0 ? ".0h" : ".5h");
				Object valueToc = mapToc.get(key);
				Object valueSs = mapSs.get(key);
				Object valueTn = mapTn.get(key);
				Object valueTp = mapTp.get(key);
				Object valueFlux = mapFlux.get(key);
				Object valuePh = mapPh.get(key);
				
				if(valueToc != null 
						&& valueSs != null 
						&& valueTn != null 
						&& valueTp != null 
						&& valueFlux != null
						&& valuePh != null) {
					predictions[index - 1] = TmsPredict.builder()
						.toc(((Number) valueToc).doubleValue())
						.ss(((Number) valueSs).doubleValue())
						.tn(((Number) valueTn).doubleValue())
						.tp(((Number) valueTp).doubleValue())
						.flux(((Number) valueFlux).doubleValue())
						.ph(((Number) valuePh).doubleValue())
						.tmsTime(now.plusMinutes(index * 30))
						.build();
					if(predictions[index - 1].getToc() > 15.0
							|| predictions[index - 1].getSs() > 10.0
							|| predictions[index - 1].getPh() > 8.5
							|| predictions[index - 1].getPh() < 5.8
							|| predictions[index - 1].getTn() > 10.0
							|| predictions[index - 1].getTp() > 0.5) {
						checkOutLier = true;
					}
						
				} else {
					System.out.println(response);
					System.err.println("예측값 누락 (" + key + ")");
					return null;
				}
			}
			if(checkOutLier) {
				logService.addOutLierLog("tms", mapPredictions.toString());
			}
			
			
			return predictions;
		} catch (NumberFormatException e) {
			System.err.println("예측값을 숫자로 변환하는 중 오류 발생: " + e.getMessage());
			return null;
		} catch (Exception e) {
			System.err.println("예측값 추출 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}

}