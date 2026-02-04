package kr.kro.prjectwwtp.config;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationFailureHandler;
import org.springframework.stereotype.Component;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.service.LoginLogService;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class Oauth2FailureHandler extends SimpleUrlAuthenticationFailureHandler {
	private final LoginLogService logService;
	
	@Value("${spring.auth2.URI}")
	private String redirectURI;

	@Override
	public void onAuthenticationFailure(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException exception) throws IOException, ServletException {
		// TODO Auto-generated method stub
		Member member = null;
		boolean loginSuccess = false;
		String userId = null;
		
		String remoteInfo = null;
		String socialAuth = null;
		String errorMsg = "Oauth2 인증 실패";
		
		try {
			// 브라우저/기기 정보 추출
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			String remoteAddr = Oauth2SuccessHandler.getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			remoteInfo = remoteAddr + ":" + remotePort;
			
			System.out.println("[Oauth2SuccessHandler] User Agent: " + userAgent);
			System.out.println("[Oauth2SuccessHandler] Remote IP:PORT: " + remoteInfo);
			
			response.sendRedirect(redirectURI);
		}catch(IOException e) {
			errorMsg = e.getMessage();
		}finally {
			// 로그인 기록 추가
			logService.addLoginLog(member, loginSuccess, userId, remoteInfo, socialAuth, errorMsg);
		}
	}

	
}
