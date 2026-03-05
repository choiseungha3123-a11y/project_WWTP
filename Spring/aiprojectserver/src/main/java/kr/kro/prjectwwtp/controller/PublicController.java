package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.CookieValue;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MultipartFile;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.PublicDTO;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.PublicRepository;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

@RestController
@RestControllerAdvice
@RequestMapping("/api/public")
@RequiredArgsConstructor
@Tag(name="PublicController", description = "OAuth2 로그인 유저들을 위한 API")
public class PublicController {
	private final PublicRepository repo;
	private PasswordEncoder encoder = new PasswordEncoder();
	
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
	
	@GetMapping("get")
	@Operation(summary="OAuth2 이용자들의 글 조회")
	@Parameter(name = "page", description= "조회할 페이지수", example = "0")
	@Parameter(name = "count", description= "페이지 별로 보여줄 메모의 수", example = "10")
	public ResponseEntity<Object> get(
			HttpServletRequest request,
			@RequestParam int page,
			@RequestParam int size) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.build();
		Pageable pageable = PageRequest.of(page, size, Sort.by("no").descending());
		
		Page<PublicDTO> data = repo.findByDeleteTimeIsNull(pageable);
		for(PublicDTO dto : data.getContent()) {
			if(dto.getMember() != null && dto.getMember().getUserName() != null)
				dto.setUserName(dto.getMember().getUserName());
			res.addData(dto);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@NoArgsConstructor
	public static class insertDTO {
		String pos;
		String content;
		String password;
		MultipartFile file;
	}
	
	@PutMapping("put")
	@Operation(summary="OAuth2 이용자들의 글 등록")
	public ResponseEntity<Object> put(
			HttpServletRequest request,
			//@CookieValue(name="jwtToken", required = true) String jwtToken,
			@CookieValue String jwtToken,
			@ModelAttribute insertDTO insert) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.build();
		String error = null;
		try {
			if(insert.pos == null || insert.content == null || insert.password == null) {
				res.setSuccess(false);
				res.setErrorMsg("정보가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			Member member = JWTUtil.parseToken(jwtToken);
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			String remoteAddr = Util.getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			String remoteInfo = remoteAddr + ":" + remotePort;
			byte[] imageData = null;
			if(insert.file != null) {
				imageData = insert.file.getBytes();
			}
			repo.save(PublicDTO.builder()
					.member(member)
					.userAgent(userAgent)
					.remoteInfo(remoteInfo)
					.pos(insert.pos)
					.content(insert.content)
					.password(encoder.encode(insert.password))
					.picture(imageData)
					.build());
		} catch(Exception e) {
			error = e.getMessage();
			res.setErrorMsg(error);
			res.setSuccess(false);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@NoArgsConstructor
	public static class modifyDTO {
		Long no;
		String pos;
		String content;
		String password;
		MultipartFile file;
	}
	@PatchMapping("patch")
	@Operation(summary="OAuth2 이용자들의 글 수정")
	public ResponseEntity<Object> patch(
			HttpServletRequest request,
			//@CookieValue(name="jwtToken", required = true) String jwtToken,
			@CookieValue String jwtToken,
			@ModelAttribute modifyDTO modify) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.build();
		String error = null;
		try {
			if(modify.no == null || (modify.pos == null && modify.content == null && modify.file == null) || modify.password == null) {
				res.setSuccess(false);
				res.setErrorMsg("정보가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			Optional<PublicDTO> opt = repo.findById(modify.no);
			if(opt.isEmpty()) {
				res.setSuccess(false);
				res.setErrorMsg("no가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			PublicDTO dto = opt.get();
			if(!encoder.matches(modify.password, dto.getPassword())) {
				res.setSuccess(false);
				res.setErrorMsg("비밀번호가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			Member member = JWTUtil.parseToken(jwtToken);
			if(member.getUserNo() != dto.getMember().getUserNo() 
					&& member.getRole() != kr.kro.prjectwwtp.domain.Role.ROLE_ADMIN) {
				res.setSuccess(false);
				res.setErrorMsg("권한이 없습니다.");
				return ResponseEntity.ok().body(res);
			}
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			String remoteAddr = Util.getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			String remoteInfo = remoteAddr + ":" + remotePort;
			byte[] imageData = null;
			if(modify.file != null) {
				imageData = modify.file.getBytes();
			}
			if(modify.pos != null)
				dto.setPos(modify.pos);
			if(modify.content != null)
				dto.setContent(modify.content);
			if(imageData != null)
				dto.setPicture(imageData);
			dto.setUserAgent(userAgent);
			dto.setRemoteInfo(remoteInfo);
			dto.setModifyTime(LocalDateTime.now());
			repo.save(dto);
		} catch(Exception e) {
			error = e.getMessage();
			res.setErrorMsg(error);
			res.setSuccess(false);
		}
		return ResponseEntity.ok().body(res);
	}
	
	@Getter
	@Setter
	@NoArgsConstructor
	public static class deleteDTO {
		Long no;
		String password;
	}
	@DeleteMapping("delete")
	@Operation(summary="OAuth2 이용자들의 글 삭제")
	public ResponseEntity<Object> delete(
			HttpServletRequest request,
			//@CookieValue(name="jwtToken", required = true) String jwtToken,
			@CookieValue String jwtToken,
			@RequestBody deleteDTO delete) {
		responseDTO res = responseDTO.builder()
				.success(true)
				.build();
		String error = null;
		try {
			if(delete.no == null || delete.password == null) {
				res.setSuccess(false);
				res.setErrorMsg("정보가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			Optional<PublicDTO> opt = repo.findById(delete.no);
			if(opt.isEmpty()) {
				res.setSuccess(false);
				res.setErrorMsg("no가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			PublicDTO dto = opt.get();
			if(!encoder.matches(delete.password, dto.getPassword())) {
				res.setSuccess(false);
				res.setErrorMsg("비밀번호가 올바르지 않습니다.");
				return ResponseEntity.ok().body(res);
			}
			Member member = JWTUtil.parseToken(jwtToken);
			if(member.getUserNo() != dto.getMember().getUserNo() 
					&& member.getRole() != kr.kro.prjectwwtp.domain.Role.ROLE_ADMIN) {
				res.setSuccess(false);
				res.setErrorMsg("권한이 없습니다.");
				return ResponseEntity.ok().body(res);
			}
			dto.setDeleteTime(LocalDateTime.now());
			repo.save(dto);
		} catch(Exception e) {
			error = e.getMessage();
			res.setErrorMsg(error);
			res.setSuccess(false);
		}
		return ResponseEntity.ok().body(res);
	}

}
