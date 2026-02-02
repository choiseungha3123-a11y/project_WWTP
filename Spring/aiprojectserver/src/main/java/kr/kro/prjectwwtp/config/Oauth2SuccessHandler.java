package kr.kro.prjectwwtp.config;

import java.io.IOException;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.service.LoginLogService;
import kr.kro.prjectwwtp.service.MemberService;
import kr.kro.prjectwwtp.service.SessionService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class Oauth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler{
	private final MemberService memberService;
	private final LoginLogService logService;
	private final SessionService sessionService;
	
	// 소셜 로그인시 주소 체크!!!!
	@Value("${spring.auth2.URI}")
	private String redirectURI;
	
	@Override
	public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
			Authentication authentication) throws IOException, ServletException {
		// TODO Auto-generated method stub
		Member member = null;
		boolean loginSuccess = false;
		String userId = null;
		
		String remoteInfo = null;
		String socialAuth = null;
		String errorMsg = null;
		
		try {
			Map<String, String> map = getUseInfo(authentication);
	
			String provider = map.get("provider");
			String name = map.get("name");
			String email = map.get("email");
			
			userId = name + "@" + provider;
			socialAuth = name + "@" + provider + "@" + email;
			
			System.out.println("OAuth2 인증 : " + socialAuth);
			
			
			member = memberService.findBySocialAuth(socialAuth);
			
			if(member != null) {
				// 기존 로그인 유저
			} else {
				// 신규 가입
				member = memberService.addSocialMember(socialAuth, userId, name);
			}
			
			// 기존 세션 만료 (동시 로그인 차단)
			System.out.println("[Oauth2SuccessHandler] Expiring previous sessions for user: " + member.getUserId());
			sessionService.expireUserSessions(member.getUserId());
	
			// JWT 생성
			String token = JWTUtil.getJWT(member);
			
			// 브라우저/기기 정보 추출
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			String remoteAddr = getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			remoteInfo = remoteAddr + ":" + remotePort;
			
			System.out.println("[Oauth2SuccessHandler] User Agent: " + userAgent);
			System.out.println("[Oauth2SuccessHandler] Remote IP:PORT: " + remoteInfo);
			
			// 새로운 세션 등록
			System.out.println("[Oauth2SuccessHandler] Registering new session for user: " + member.getUserId());
			sessionService.registerNewSession(member.getUserId(), token, userAgent, remoteInfo);
			
			//System.out.println("token : " + token);
			// Cookie에 jwt 추가
			Cookie cookie = new Cookie("jwtToken", token.replaceAll(JWTUtil.prefix, ""));
			cookie.setHttpOnly(true);	// JS에서 접근 못 하게
			cookie.setSecure(false);	// HTTPS에서만 동작
			cookie.setPath("/");
			cookie.setMaxAge(60 * 60);		// 60초 * 60 = 1시간
			response.addCookie(cookie);
		
			loginSuccess = true;
			// 로그인 후 초기 페이지
			response.sendRedirect(redirectURI);
		}catch(IOException e) {
			errorMsg = e.getMessage();
		}finally {
			// 로그인 기록 추가
			logService.addLoginLog(member, loginSuccess, userId, remoteInfo, socialAuth, errorMsg);
		}
	}

	@SuppressWarnings("unchecked")
	Map<String, String> getUseInfo(Authentication authentication) {
		OAuth2AuthenticationToken oAuth2Token = (OAuth2AuthenticationToken)authentication;
		
		String provider = oAuth2Token.getAuthorizedClientRegistrationId();
		//System.out.println("[OAuth2SuccessHandler]provider: " + provider);
		
		OAuth2User user = (OAuth2User)oAuth2Token.getPrincipal();
		//String email = "unknown";
		String name = "unknown";
		// 로그인 방법별 데이터 구성
		if(provider.equalsIgnoreCase("naver")) {			// naver
			Map<String, Object> response = (Map<String, Object>)user.getAttribute("response");
			name = (String)response.get("name");
			//email = (String)response.get("email");
		} else if(provider.equalsIgnoreCase("google")) {	// google
			name = (String)user.getAttributes().get("name");
			//email = (String)user.getAttributes().get("email");
		} else if(provider.equalsIgnoreCase("kakao")) {		// kakao
			Map<String, String> properties = (Map<String, String>)user.getAttributes().get("properties");  
			name = properties.get("nickname");
		}
		//System.out.println("[OAuth2SuccessHandler]email: " + email);
		return Map.of("provider", provider, "name", name);
	}
	
	/**
	 * 클라이언트의 실제 IP 주소 추출
	 * 프록시 환경에서도 올바른 IP를 가져오도록 처리
	 */
	private String getRemoteAddress(HttpServletRequest request) {
		String ip = request.getHeader("X-Forwarded-For");
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getHeader("Proxy-Client-IP");
		}
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getHeader("WL-Proxy-Client-IP");
		}
		if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
			ip = request.getRemoteAddr();
		}
		// X-Forwarded-For가 여러 IP를 포함할 수 있으므로 첫 번째만 사용
		if (ip != null && ip.contains(",")) {
			ip = ip.split(",")[0].trim();
		}
		return ip;
	}
}
