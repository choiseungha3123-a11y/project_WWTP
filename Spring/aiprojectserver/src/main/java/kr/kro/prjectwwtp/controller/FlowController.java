package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

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
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.FlowImputate;
import kr.kro.prjectwwtp.domain.FlowOrigin;
import kr.kro.prjectwwtp.domain.FlowPredict;
import kr.kro.prjectwwtp.domain.Input;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.WeatherDTO;
import kr.kro.prjectwwtp.domain.fastApiResponseDTO;
import kr.kro.prjectwwtp.domain.predictIn;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.FastApiService;
import kr.kro.prjectwwtp.service.FlowService;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/flowOrigin")
@RequiredArgsConstructor
@Tag(name="FlowOriginController", description = "유량 수치 처리 API")
public class FlowController {
	private final LogService logService;
	private final FlowService flowService;
	private final WeatherService weatherService;
	private final FastApiService apiService;
	
	@Value("${predict.enable}")
	private boolean enablePredict;
	
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
			saveCount = flowService.saveFromCsv(file);
			res.addData("saveCount : " + saveCount);
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
			logService.addErrorLog("FlowController.java", "postFlowOriginUpload()", e.getMessage());
		} finally {
			logService.addFlowLog(member, "list", saveCount, errorMsg);
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
			List<FlowOrigin> list = flowService.getFlowOriginListByDate(time);
			for(FlowOrigin t : list) {
				res.addData(t);
			}
			listSize = list.size();
		} catch (Exception e) {
			res.setSuccess(false);
			errorMsg = e.getMessage();
			res.setErrorMsg(errorMsg);
			logService.addErrorLog("FlowController.java", "getFlowOriginList()", e.getMessage());
		} finally {
			logService.addFlowLog(member, "list", listSize, errorMsg);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/test")
	@Scheduled(cron = "${scheduler.predict.cron}", zone="${spring.timezone}")
	public ResponseEntity<Object>  getFlowPredict() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		if(!enablePredict) {
			res.setSuccess(false);
			res.setErrorMsg("예측 서비스가 비활성화 되었습니다.");
			return ResponseEntity.ok().body(res);
		}
		try {
			LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
			LocalDateTime fakeNow = flowService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute())
									.withSecond(0)
									.withNano(0);
			List<FlowImputate> flowList = flowService.getFlowImputateListByDate(fakeNow);
			System.out.println("flowList : " + flowList.size());
			List<WeatherDTO> aws368 = weatherService.findWeatherDTOByStnAndLogTimeBetween(368, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws368 : " + aws368.size());
			List<WeatherDTO> aws541 = weatherService.findWeatherDTOByStnAndLogTimeBetween(541, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws541 : " + aws541.size());
			List<WeatherDTO> aws569 = weatherService.findWeatherDTOByStnAndLogTimeBetween(569, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws569 : " + aws569.size());
			requestFlow(now, aws368, aws541, aws569, flowList);
		} catch (Exception e) {
			logService.addErrorLog("FlowController.java", "getFlowPredict()", e.getMessage());
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());;
		}
		return ResponseEntity.ok().body(res);
	}

	public void requestFlow(LocalDateTime now, List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<FlowImputate>flowList) {
		String errorMsg = null;
		int predictSize = 0;
		try {
			Input<FlowImputate> input = new Input<>(aws368, aws541, aws569, flowList);
			predictIn<FlowImputate> pIn = new predictIn<>(input);
			System.out.println("requestFlow start");
			fastApiResponseDTO response = apiService.getPredict("/predict/flow", pIn);
			System.out.println("response = " + response.isOk());
			if(response.isOk()) {
				FlowPredict[] predictions = extractPredictions(now, response);
				predictSize = predictions.length;
				//System.out.println("예측값 (0.5h~12.0h): " + java.util.Arrays.toString(predictions));
				flowService.savePredictList(predictions);
			}
			else {
				errorMsg = response.getError();
				System.out.println("getPredict fail : " + errorMsg);
			}
		}catch(Exception e) {
			e.printStackTrace();
			errorMsg = e.getMessage();
			logService.addErrorLog("FlowController.java", "requestFlow()", e.getMessage());
		}
		finally {
			logService.addFlowLog(null, "predict", predictSize, errorMsg);
		}
	}

	@GetMapping("/flowList")
	@Operation(summary="어제부터의 실시간 정보와 내일까지의 예상 정보를 요청 (Java 코드 처리)", 
	           description = "결측/이상 값을 처리한 데이터를 조회합니다. 중복 제거는 Java 코드로 처리합니다.")
	public ResponseEntity<Object> getFlowList() {
		responseDTO res = responseDTO.builder()
				.success(true)
				.errorMsg(null)
				.build();
		LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
		LocalDateTime end = now.plusDays(1).minusMinutes(1);
		List<FlowPredict> list = flowService.findPredictList(now, end);
		res.addData(list);
		
		return ResponseEntity.ok().body(res);
	}
	
	/**
	 * FastAPI 응답에서 predictions 값을 1h~12h 순으로 double 배열로 추출
	 * @param response FastAPI 응답 DTO
	 * @return predictions 배열 (크기: 12), 추출 실패 시 null
	 */
	private FlowPredict[] extractPredictions(LocalDateTime now, fastApiResponseDTO response) {
		int predictSize = 24;
		FlowPredict[] predictions = new FlowPredict[predictSize];
		ObjectMapper mapper = new ObjectMapper();
		
		try {
			if(response == null || response.getOutput() == null) {
				System.err.println("응답 또는 output이 null입니다");
				return null;
			}
			
			Map<String, Object> mapOutput = response.getOutput();
			Map<String, Object> mapPredictions = mapper.convertValue(mapOutput.get("predictions"),new TypeReference<>() {});
			// output에서 predictions 데이터 추출
			for(int index = 1; index <= predictSize; index++) {
				String key = index/2 + (index % 2 == 0 ? ".0h" : ".5h");
				Object value = mapPredictions.get(key);
				
				if(value != null) {
					predictions[index - 1] = FlowPredict.builder()
							.flowValue(((Number) value).doubleValue())
							.flowTime(now.plusMinutes(index * 30))
							.createTime(now)
							.build();
				} else {
					System.out.println(response);
					System.err.println("예측값 누락 (" + key + ")");
					return null;
				}
			}
			
			return predictions;
		} catch (NumberFormatException e) {
			System.err.println("예측값을 숫자로 변환하는 중 오류 발생: " + e.getMessage());
			logService.addErrorLog("FlowController.java", "extractPredictions()", e.getMessage());
			return null;
		} catch (Exception e) {
			System.err.println("예측값 추출 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}

}