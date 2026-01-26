package kr.kro.prjectwwtp.controller;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import io.swagger.v3.oas.annotations.parameters.RequestBody;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.LoginLog;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.LoginLogRepository;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Controller
@RequiredArgsConstructor
public class SecurityController {
	private final MemberRepository memberRepo;
	private final LoginLogRepository logRepo;
	private PasswordEncoder encoder = new PasswordEncoder();
	// 접근 권한 오류 페이지
	@GetMapping("/system/accessDenied")
	public void accessDenied() {}
	
	// 로그인 처리
	@GetMapping("system/login")
	public String login() {
		return "system/login.html";
	}
	
	// 로그아웃 처리
	@GetMapping("/system/logout")
	public void logout() {}
	
	// 회원 가입 처리
	@GetMapping("/system/addMember")
	public void addMember() {}
	
	// 관리자 페이지(사용할지 안할지 모름)
	@GetMapping("/admin/adminPage")
	public void adminPage() {}
	
//	// IP 체크를 위한 임시 페이지
//	@GetMapping("/system/check-ip")
//    @ResponseBody // 이 어노테이션이 있어야 IP 주소를 템플릿 파일로 오해하지 않습니다.
//    public String getIp(HttpServletRequest request) {
//        String ip = request.getRemoteAddr();
//        int port = request.getRemotePort();
//        return ip+":"+port; // 이제 "10.125.121.176" 문자열이 화면에 그대로 출력됩니다.
//    }
	
	
	
}
