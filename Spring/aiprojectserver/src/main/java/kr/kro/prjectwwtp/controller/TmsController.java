package kr.kro.prjectwwtp.controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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
import kr.kro.prjectwwtp.domain.Input;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.domain.TmsPredict;
import kr.kro.prjectwwtp.domain.WeatherDTO;
import kr.kro.prjectwwtp.domain.fastApiResponseDTO;
import kr.kro.prjectwwtp.domain.predictIn;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.FastApiService;
import kr.kro.prjectwwtp.service.FlowService;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.service.TmsService;
import kr.kro.prjectwwtp.service.WeatherService;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
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
			logService.addErrorLog("TmsController.java", "postTmsOriginUpload()", e.getMessage());
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
			logService.addErrorLog("TmsController.java", "getTmsOriginList()", e.getMessage());
		} finally {
			logService.addTmsLog(member, "list", listSize, errorMsg);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@Scheduled(cron = "${scheduler.fakeday.cron}", zone="${spring.timezone}")
	//@GetMapping("/makeFakeNow")
	public void makeFakeDate() {
		try {
			//System.out.println("makeFakeDate");
			LocalDateTime now = LocalDateTime.now();
			LocalDateTime fakeTmeNow = tmsService.getFakeNow()
										.withHour(now.getHour())
										.withMinute(now.getMinute());
			//System.out.println("fakeTmeNow : " + fakeTmeNow);
			
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
			//System.out.println("fakeFlowNow : " + fakeFlowNow);
					
			// 조회할 날짜(fakeTmeNow를 기준으로 이전 날짜와 해당 날짜의 보간 데이터 구성
			if(!flowService.existsByFlowTime(fakeFlowNow)) {
				List<FlowImputate> list = flowService.imputate(fakeFlowNow);
				flowService.saveFlowImputateList(list);
			}
			if(!flowService.existsByFlowTime(fakeFlowNow.minusDays(1))) {
				List<FlowImputate> list = flowService.imputate(fakeFlowNow.minusDays(1));
				flowService.saveFlowImputateList(list);
			}
		}catch(Exception e) {
			e.printStackTrace();
			logService.addErrorLog("TmsController.java", "makeFakeDate()", e.getMessage());
		}
	}
	
	@GetMapping("/test")
	@Scheduled(cron = "${scheduler.predict.cron}", zone="${spring.timezone}")
	public ResponseEntity<Object>   getTmsPredict() {
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
			LocalDateTime fakeNow = tmsService.getFakeNow()
									.withHour(now.getHour())
									.withMinute(now.getMinute())
									.withSecond(0)
									.withNano(0);
			List<TmsImputate> tmsList = tmsService.getTmsImputateListByDate(fakeNow);
			System.out.println("tmsList : " + tmsList.size());
			List<WeatherDTO> aws368 = weatherService.findWeatherDTOByStnAndLogTimeBetween(368, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws368 : " + aws368.size());
			List<WeatherDTO> aws541 = weatherService.findWeatherDTOByStnAndLogTimeBetween(541, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws541 : " + aws541.size());
			List<WeatherDTO> aws569 = weatherService.findWeatherDTOByStnAndLogTimeBetween(569, fakeNow.minusDays(1).plusMinutes(1), fakeNow);
			System.out.println("aws569 : " + aws569.size());
			
			TmsPredict[] predictions = requestTms(now, aws368, aws541, aws569, tmsList);
			List<TmsImputate> tmsListReal = tmsService.getTmsImputateListBetwwen(fakeNow, fakeNow.plusHours(12));
			
			// 데이터 확인을 위해 데이터를 임시로 cvs로 저장
			// tmsList, aws368, aws541, asw569, predictions, 실제 이 기간의 tms 실측값
			if(predictions != null)
				saveToCsv(now, tmsList, aws368, aws541, aws569, predictions, tmsListReal, fakeNow);
								
		} catch (Exception e) {
			e.printStackTrace();
			logService.addErrorLog("TmsController.java", "getTmsPredict()", e.getMessage());
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}
	
	public TmsPredict[] requestTms(LocalDateTime now, List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<TmsImputate>tmsList) {
		String errorMsg = null;
		int predictSize = 0;
		try {
			Input<TmsImputate> input = new Input<>(aws368, aws541, aws569, tmsList);
			predictIn<TmsImputate> pIn = new predictIn<>(input);
			System.out.println("requestTms start");
			fastApiResponseDTO response = apiService.getPredict("/predict/tms", pIn);
			System.out.println("response = " + response.isOk());
			if(response.isOk()) {
				TmsPredict[] predictions = extractPredictions(now, response);
				predictSize = predictions.length;
				//System.out.println("예측값 (0.5h~12.0h): " + java.util.Arrays.toString(predictions));
				tmsService.savePredictList(predictions);
				return predictions;
			}
			else {
				errorMsg = response.getError();
				System.out.println("getPredict fail : " + errorMsg);
			}
		}catch(Exception e) {
			e.printStackTrace();
			errorMsg = e.getMessage();
			logService.addErrorLog("TmsController.java", "requestTms()", e.getMessage());
		}
		finally {
			logService.addTmsLog(null, "predict", predictSize, errorMsg);
		}
		return null;
	}
	
	private void saveToCsv(
			LocalDateTime now,
			List<TmsImputate> tmsList, 
			List<WeatherDTO> aws368, 
			List<WeatherDTO> aws541, 
			List<WeatherDTO> aws569, 
			TmsPredict[] predictions, 
			List<TmsImputate> tmsListReal,
			LocalDateTime fakeNow) {
		try {
			String fileName = "요청내용" + now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm00")) + ".csv";
			File file = Util.resolveFilePath(fileName);
			
			String data = "";
			data += "SYS_TIME, TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU, ASW368, TA, RN_15m, RN_60m, RN_12H, RN_DAY, HM, TD, distance, ASW541, TA, RN_15m, RN_60m, RN_12H, RN_DAY, HM, TD, distance, ASW569, TA, RN_15m, RN_60m, RN_12H, RN_DAY, HM, TD, distance\r\n";
			for(TmsImputate tms : tmsList) {
				LocalDateTime time = tms.getTmsTime();
				data += "'" + tms.getStrtime() + "', ";
				data += tms.getToc() + ", ";
				data += tms.getPh() + ", ";
				data += tms.getSs() + ", ";
				data += tms.getFlux() + ", ";
				data += tms.getTn() + ", ";
				data += tms.getTp() + ", ";
				
				WeatherDTO weather368 = getWeatherDto(time, aws569);
				if(weather368 != null) {
					data += ", ";
					data += weather368.getTa() + ", ";
					data += weather368.getRn15m() + ", ";
					data += weather368.getRn60m() + ", ";
					data += weather368.getRn12h() + ", ";
					data += weather368.getRnday() + ", ";
					data += weather368.getHm() + ", ";
					data += weather368.getTd() + ", ";
					data += weather368.getDistance() + ", ";
				} else {
					data += ", , , , , , , , , ";
				}
				
				WeatherDTO weather541 = getWeatherDto(time, aws541);
				if(weather541 != null) {
					data += ", ";
					data += weather541.getTa() + ", ";
					data += weather541.getRn15m() + ", ";
					data += weather541.getRn60m() + ", ";
					data += weather541.getRn12h() + ", ";
					data += weather541.getRnday() + ", ";
					data += weather541.getHm() + ", ";
					data += weather541.getTd() + ", ";
					data += weather541.getDistance() + ", ";
				} else {
					data += ", , , , , , , , , ";
				}
				
				WeatherDTO weather569 = getWeatherDto(time, aws569);
				if(weather569 != null) {
					data += ", ";
					data += weather569.getTa() + ", ";
					data += weather569.getRn15m() + ", ";
					data += weather569.getRn60m() + ", ";
					data += weather569.getRn12h() + ", ";
					data += weather569.getRnday() + ", ";
					data += weather569.getHm() + ", ";
					data += weather569.getTd() + ", ";
					data += weather569.getDistance() + ", \r\n";
				} else {
					data += ", , , , , , , , , \r\n";
				}
			}
			
			// UTF-8 인코딩으로 파일 작성
			try (BufferedWriter bw = new BufferedWriter(
					new OutputStreamWriter(new FileOutputStream(file.getAbsolutePath()), "UTF-8"))) {
				bw.write(data);
				bw.flush();
				bw.close();
			}
			
			String fileName2 = "예측내용" + now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm00")) + ".csv";
			File file2 = Util.resolveFilePath(fileName2);
			
			String data2 = "";
			data2 += "SYS_TIME, TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU, Origin Data, TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU \r\n";
			for(TmsPredict tms : predictions) {
				LocalDateTime time = tms.getTmsTime();
				data2 += "'" + tms.getTmsTime() + "', ";
				data2 += tms.getToc() + ", ";
				data2 += tms.getPh() + ", ";
				data2 += tms.getSs() + ", ";
				data2 += tms.getFlux() + ", ";
				data2 += tms.getTn() + ", ";
				data2 += tms.getTp() + ", ";
				
				TmsImputate tmsReal =  getTmsReal(time, tmsListReal);
				if(tmsReal != null) {
					data2 += ", ";
					data2 += tmsReal.getToc() + ", ";
					data2 += tmsReal.getPh() + ", ";
					data2 += tmsReal.getSs() + ", ";
					data2 += tmsReal.getFlux() + ", ";
					data2 += tmsReal.getTn() + ", ";
					data2 += tmsReal.getTp() + "\r\n";
				} else  {
					data2 += ", , , , , , , , \r\n";
				}
			}
				
			// UTF-8 인코딩으로 파일 작성
			try (BufferedWriter bw = new BufferedWriter(
					new OutputStreamWriter(new FileOutputStream(file2.getAbsolutePath()), "UTF-8"))) {
				bw.write(data2);
				bw.flush();
				bw.close();
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private WeatherDTO getWeatherDto(LocalDateTime time, List<WeatherDTO> list) {
		if(list == null || time == null) {
			return null;
		}
		String strTime = time.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
		for(WeatherDTO weather : list) {
			if(weather.getTime().equals(strTime)) {
				return weather;
			}
		}
		return null;
	}
	
	private TmsImputate getTmsReal(LocalDateTime time, List<TmsImputate> list) {
		if(list == null || time == null) {
			return null;
		}
		DateTimeFormatter pattern = DateTimeFormatter.ofPattern("HHmmss");
		for(TmsImputate tms : list) {
			if(tms.getTmsTime().format(pattern).equals(time.format(pattern))) {
				return tms;
			}
		}
		return null;
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
	
	/**
	 * FastAPI 응답에서 predictions 값을 1h~12h 순으로 double 배열로 추출
	 * @param response FastAPI 응답 DTO
	 * @return predictions 배열 (크기: 12), 추출 실패 시 null
	 */
	private TmsPredict[] extractPredictions(LocalDateTime now, fastApiResponseDTO response) {
		int predictSize = 24;
		boolean checkOutLierToc = false;
		boolean checkOutLierSs = false;
		boolean checkOutLierPh = false;
		boolean checkOutLierTn = false;
		boolean checkOutLierTp = false;
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
						.createTime(now)
						.build();
					if(predictions[index - 1].getToc() > 15.0)
						checkOutLierToc = true;
					if(predictions[index - 1].getSs() > 10.0)
						checkOutLierSs = true;
					if(predictions[index - 1].getPh() > 8.5
							|| predictions[index - 1].getPh() < 5.8)
						checkOutLierPh = true;
					if(predictions[index - 1].getTn() > 10.0)
						checkOutLierTn = true;
					if(predictions[index - 1].getTp() > 0.5)
						checkOutLierTp = true;
						
				} else {
					System.out.println(response);
					System.err.println("예측값 누락 (" + key + ")");
					return null;
				}
			}
			if(checkOutLierToc || checkOutLierSs || checkOutLierPh || checkOutLierTn || checkOutLierTp) {
				String type = "tms(";
				if(checkOutLierToc)
					type += "toc ";
				if(checkOutLierSs)
					type += "ss ";
				if(checkOutLierPh)
					type += "ph ";
				if(checkOutLierTn)
					type += "tn ";
				if(checkOutLierTp)
					type += "tp";
				type += ")";
				logService.addOutLierLog(type, mapPredictions.toString());
			}
			
			
			return predictions;
		} catch (NumberFormatException e) {
			System.err.println("예측값을 숫자로 변환하는 중 오류 발생: " + e.getMessage());
			logService.addErrorLog("TmsController.java", "extractPredictions()", e.getMessage());
			return null;
		} catch (Exception e) {
			System.err.println("예측값 추출 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}

}