package kr.kro.prjectwwtp.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

import lombok.RequiredArgsConstructor;

@Controller
@RequiredArgsConstructor
public class SecurityController {
	
	// 접근 권한 오류 페이지
	@GetMapping("/system/accessDenied")
	public void accessDenied() {}
	
//	// IP 체크를 위한 임시 페이지
//	@GetMapping("/system/check-ip")
//    @ResponseBody // 이 어노테이션이 있어야 IP 주소를 템플릿 파일로 오해하지 않습니다.
//    public String getIp(HttpServletRequest request) {
//        String ip = request.getRemoteAddr();
//        int port = request.getRemotePort();
//        return ip+":"+port; // 이제 "10.125.121.176" 문자열이 화면에 그대로 출력됩니다.
//    }
	
	
	
}
