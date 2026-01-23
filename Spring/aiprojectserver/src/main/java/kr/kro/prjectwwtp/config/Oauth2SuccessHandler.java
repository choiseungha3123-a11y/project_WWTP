package kr.kro.prjectwwtp.config;

import java.io.IOException;
import java.util.Map;

import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class Oauth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler{
	
	@Override
	public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
			Authentication authentication) throws IOException, ServletException {
		// TODO Auto-generated method stub
		// JWT 생성
		//String token = JWTUtil.getJWT(user);
		//System.out.println("token : " + token);
		// Cookie에 jwt 추가
		//Cookie cookie = new Cookie("jwtToken", token.replaceAll(JWTUtil.prefix, ""));
		//cookie.setHttpOnly(true);	// JS에서 접근 못 하게
		//cookie.setSecure(false);	// HTTPS에서만 동작
		//cookie.setPath("/");
		//cookie.setMaxAge(60 * 60);		// 60초 * 60 = 1시간
		//response.addCookie(cookie);
		
		try {
			// 로그인 후 초기 페이지
			response.sendRedirect("http://localhost:3000");
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
}
