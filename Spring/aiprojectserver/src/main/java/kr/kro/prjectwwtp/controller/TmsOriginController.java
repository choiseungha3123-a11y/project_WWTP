package kr.kro.prjectwwtp.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
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

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.TmsOriginService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RestController
@RestControllerAdvice
@RequestMapping("/api/tmsOrigin")
@RequiredArgsConstructor
@Tag(name="TmsOriginController", description = "TMS Origin API")
public class TmsOriginController {
	private final TmsOriginService tmsOriginService;
	
	@Value("${spring.FastAPI.URI}")
	private String fastAPIURI;
	
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
			int saveCount = tmsOriginService.saveFromCsv(file);
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
		@ApiResponse(responseCode = "201", description = "dataList[]", content = @Content(mediaType = "application/json", schema = @Schema(implementation = TmsOrigin.class)))
	})
	public ResponseEntity<Object> getTmsOriginList(
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
			List<TmsOrigin> list = tmsOriginService.getTmsOriginListByDate(time);
			for(TmsOrigin t : list) {
				res.addData(t);
			}
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}
	
	@GetMapping("/imputate")
	@Operation(summary="실제 측정 데이터 조회", description = "저장된 실제 측정 데이터를 조회합니다.")
	@Parameter(name = "time", description= "조회날짜(yyyyMMdd)", example = "20240101")
	public ResponseEntity<Object> postTmsOriginImputate(
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
			/*
			// CSV 파일 체크
			String csvFilePath = "Downloads/imputated_data_" + time + ".csv";
			
			List<TmsOrigin> list = tmsOriginService.loadFromCsv(csvFilePath);
			if(list == null || list.size() == 0) {
				// CSV 파일이 없으면 보간 수행
				list = tmsOriginService.imputate(time);
				// 보간된 데이터를 CSV 파일로 저장
				tmsOriginService.saveToCsv(list, csvFilePath);
			} 
			*/
			List<TmsImputate> list = tmsOriginService.getTmsImputateListByDate(time);
			
			if(list == null || list.size() == 0) {
				// 보간된 데이터가 없으면 보간 수행
				list = tmsOriginService.imputate(time);
				tmsOriginService.saveTmsImputateList(list);
			}
			
//			String csvFilePath = "Downloads/imputated_data_" + time + ".csv";
//			tmsOriginService.saveToCsv(list, csvFilePath);
			// list를 파이썬 faseapi로 보내서 예측 수행 후 결과 저장
			
			for(TmsImputate t : list) {
				res.addData(t);
			}
			
		} catch (Exception e) {
			res.setSuccess(false);
			res.setErrorMsg(e.getMessage());
		}
		return ResponseEntity.ok().body(res);
	}

}