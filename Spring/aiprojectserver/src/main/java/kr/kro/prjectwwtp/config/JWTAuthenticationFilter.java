package kr.kro.prjectwwtp.config;

import java.io.IOException;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.service.AccessLogService;
import kr.kro.prjectwwtp.service.LoginLogService;
import kr.kro.prjectwwtp.service.SessionService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthenticationFilter extends UsernamePasswordAuthenticationFilter {
	private final AuthenticationManager authenticationManager;
	private final TokenBlacklistManager tokenBlacklistManager;
	private final SessionService sessionService;
	private final AccessLogService logService;
	private final LoginLogService loginService;
	
	@Override
	public Authentication attemptAuthentication(HttpServletRequest request, HttpServletResponse response)
			throws AuthenticationException {
		Member member = null;
		String method = request.getMethod();
		String userAgent = request.getHeader("User-Agent");
		if (userAgent == null) {
			userAgent = "Unknown";
		}
		String remoteAddr = getRemoteAddress(request);
		int remotePort = request.getRemotePort();
		String remoteInfo = remoteAddr + ":" + remotePort;
		
		String requestURI = request.getRequestURI();
		String errorMsg = null;
		
		System.out.println("\n========== [JWTAuthenticationFilter] attemptAuthentication START ==========");
		System.out.println("[JWTAuthenticationFilter] Request Method: " + method);
		System.out.println("[JWTAuthenticationFilter] Request URI: " + requestURI);
		
		try {
			ObjectMapper mapper = new ObjectMapper();
			member = mapper.readValue(request.getInputStream(), Member.class);
			
			if (member == null) {
				System.out.println("[JWTAuthenticationFilter] Member is null");
				throw new AuthenticationException("회원 정보가 없습니다") {};
			}
			
			System.out.println("[JWTAuthenticationFilter] Attempting authentication for: " + member.getUserId());
			UsernamePasswordAuthenticationToken authToken = 
				new UsernamePasswordAuthenticationToken(member.getUserId(), member.getPassword());
			return authenticationManager.authenticate(authToken);
		} catch (IOException e) {
			System.out.println("[JWTAuthenticationFilter] IOException: " + e.getMessage());
			errorMsg = e.getMessage();
			throw new AuthenticationException("요청 처리 중 오류가 발생했습니다") {};
		} finally {
			logService.addAccessLog(member, userAgent, remoteInfo, method, requestURI, errorMsg);
		}
	}
	
	@Override
	protected void successfulAuthentication(HttpServletRequest request, HttpServletResponse response, 
			FilterChain chain, Authentication authResult) throws IOException, ServletException {
		System.out.println("[JWTAuthenticationFilter] successfulAuthentication - Login successful!");
		Member member = null;
		boolean loginSuccess = false;
		String userId = null;
		
		String remoteInfo = null;
		String errorMsg = null;
		try {
			SecurityUser user = (SecurityUser) authResult.getPrincipal();
			member = user.getMember();
			userId = member.getUserId();
			
			// 브라우저/기기 정보 추출
			String userAgent = request.getHeader("User-Agent");
			if (userAgent == null) {
				userAgent = "Unknown";
			}
			
			// Remote IP 및 PORT 정보 추출
			String remoteAddr = getRemoteAddress(request);
			int remotePort = request.getRemotePort();
			remoteInfo = remoteAddr + ":" + remotePort;
			
			System.out.println("[JWTAuthenticationFilter] User Agent: " + userAgent);
			System.out.println("[JWTAuthenticationFilter] Remote IP:PORT: " + remoteInfo);
			
			// JWT 토큰 생성
			String token = JWTUtil.getJWT(member);
			System.out.println("token : " + token);
			System.out.println("[JWTAuthenticationFilter] Token generated for: " + userId);
			
			// 기존 토큰 무효화 및 새 토큰 등록 (다른 브라우저/기기의 로그인 제거)
			System.out.println("[JWTAuthenticationFilter] Registering new login (previous browser/device logins will be invalidated)");
			tokenBlacklistManager.registerNewToken(userId, token, userAgent, remoteInfo);
			
			// 기존 세션 만료 (동시 로그인 차단)
			System.out.println("[JWTAuthenticationFilter] Expiring previous sessions for user: " + userId);
			sessionService.expireUserSessions(userId);
			
			// 새로운 세션 등록
			System.out.println("[JWTAuthenticationFilter] Registering new session for user: " + userId);
			sessionService.registerNewSession(userId, token, userAgent, remoteInfo);
			
			// 응답 헤더에 토큰 추가
			response.addHeader(HttpHeaders.AUTHORIZATION, token);
			response.setStatus(HttpStatus.OK.value());
			response.setContentType(MediaType.APPLICATION_JSON_VALUE);
			response.setCharacterEncoding("UTF-8");
			
			// 응답 본문에 JSON 형식으로 토큰 반환
			responseDTO res = responseDTO.builder()
				.success(true)
				.dataSize(1)
				.build();
			res.addData(token);
			
			String responseBody = new ObjectMapper().writeValueAsString(res);
			response.getWriter().write(responseBody);
			response.getWriter().flush();
			System.out.println("========== [JWTAuthenticationFilter] attemptAuthentication END (SUCCESS) ==========\n");
			loginSuccess = true;
		}catch(Exception e) {
			errorMsg = e.getMessage();
		}finally {
			// 접속 로그 기록
			loginService.addLoginLog(member, loginSuccess, userId, remoteInfo, null, errorMsg);
		}
	}
	
	@Override
	protected void unsuccessfulAuthentication(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException failed) throws IOException, ServletException {
		response.setStatus(HttpStatus.UNAUTHORIZED.value());
		response.setContentType(MediaType.APPLICATION_JSON_VALUE);
		response.setCharacterEncoding("UTF-8");
		
		String remoteAddr = getRemoteAddress(request);
		int remotePort = request.getRemotePort();
		String remoteInfo = remoteAddr + ":" + remotePort;
		String errorMsg = "로그인에 실패했습니다. 아이디 또는 비밀번호를 확인하세요.";
		String userId = null;
		
		responseDTO res = responseDTO.builder()
			.success(false)
			.errorMsg(errorMsg)
			.build();
		
		String responseBody = new ObjectMapper().writeValueAsString(res);
		response.getWriter().write(responseBody);
		response.getWriter().flush();
		loginService.addLoginLog(null, false, userId, remoteInfo, null, errorMsg);
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
