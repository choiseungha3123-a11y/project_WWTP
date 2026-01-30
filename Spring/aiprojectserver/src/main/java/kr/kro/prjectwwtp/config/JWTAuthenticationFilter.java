package kr.kro.prjectwwtp.config;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

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
import kr.kro.prjectwwtp.service.SessionService;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthenticationFilter extends UsernamePasswordAuthenticationFilter {
	private final AuthenticationManager authenticationManager;
	private final TokenBlacklistManager tokenBlacklistManager;
	private final SessionService sessionService;
	private final AccessLogService logService;
	
	@Override
	public Authentication attemptAuthentication(HttpServletRequest request, HttpServletResponse response)
			throws AuthenticationException {
		System.out.println("\n========== [JWTAuthenticationFilter] attemptAuthentication START ==========");
		System.out.println("[JWTAuthenticationFilter] Request Method: " + request.getMethod());
		System.out.println("[JWTAuthenticationFilter] Request URI: " + request.getRequestURI());
		System.out.println("[JWTAuthenticationFilter] Content-Type: " + request.getContentType());
		System.out.println("[JWTAuthenticationFilter] Content-Length: " + request.getContentLength());
		
		byte[] requestBodyBytes = null;
		String requestBodyStr = null;
		
		try {
			// 요청 본문을 바이트 배열로 읽기
			requestBodyBytes = request.getInputStream().readAllBytes();
			requestBodyStr = new String(requestBodyBytes, StandardCharsets.UTF_8);
			
			System.out.println("[JWTAuthenticationFilter] Request Body (String): " + requestBodyStr);
			System.out.println("[JWTAuthenticationFilter] Request Body (Length): " + requestBodyStr.length() + " chars");
			System.out.println("[JWTAuthenticationFilter] Request Body (Bytes): " + requestBodyBytes.length + " bytes");
			
			// 요청 본문의 각 문자 확인 (특수 문자 체크)
			System.out.println("[JWTAuthenticationFilter] Request Body (Hex): " + bytesToHex(requestBodyBytes));
			
			if (requestBodyStr.isEmpty()) {
				System.out.println("[ERROR] [JWTAuthenticationFilter] 요청 본문이 비어있습니다!");
				throw new AuthenticationException("요청 본문이 비어있습니다") {};
			}
			
			// JSON 유효성 검사
			requestBodyStr = requestBodyStr.trim();
			if (!requestBodyStr.startsWith("{") || !requestBodyStr.endsWith("}")) {
				System.out.println("[ERROR] [JWTAuthenticationFilter] JSON 형식이 올바르지 않습니다!");
				System.out.println("  - 첫 글자: '" + (requestBodyStr.length() > 0 ? requestBodyStr.charAt(0) : "없음") + "'");
				System.out.println("  - 마지막 글자: '" + (requestBodyStr.length() > 0 ? requestBodyStr.charAt(requestBodyStr.length()-1) : "없음") + "'");
				throw new AuthenticationException("JSON 형식이 올바르지 않습니다") {};
			}
			
			ObjectMapper mapper = new ObjectMapper();
			Member member = mapper.readValue(requestBodyStr, Member.class);
			
			if (member == null) {
				System.out.println("[JWTAuthenticationFilter] ERROR: Member is null");
				throw new AuthenticationException("회원 정보가 없습니다") {};
			}
			
			System.out.println("[JWTAuthenticationFilter] Parsed Member - UserId: " + member.getUserId());
			System.out.println("[JWTAuthenticationFilter] Attempting authentication for: " + member.getUserId());
			UsernamePasswordAuthenticationToken authToken = 
				new UsernamePasswordAuthenticationToken(member.getUserId(), member.getPassword());
			return authenticationManager.authenticate(authToken);
		} catch (IOException e) {
			System.out.println("\n=============== [ERROR] [JWTAuthenticationFilter] IOException 발생! ===============");
			System.out.println("[에러 메시지]: " + e.getMessage());
			System.out.println("[에러 타입]: " + e.getClass().getName());
			System.out.println("[에러 원인]: " + (e.getCause() != null ? e.getCause().toString() : "없음"));
			
			if (requestBodyStr != null) {
				System.out.println("\n[요청 본문 정보]");
				System.out.println("  - String 형식: " + requestBodyStr);
				System.out.println("  - 길이: " + requestBodyStr.length() + " chars");
				System.out.println("  - Hex: " + bytesToHex(requestBodyBytes));
				
				// 각 라인별 분석
				String[] lines = requestBodyStr.split("\n");
				System.out.println("  - 라인 수: " + lines.length);
				for (int i = 0; i < lines.length; i++) {
					System.out.println("  - [라인 " + (i+1) + "]: " + lines[i]);
				}
			}
			
			System.out.println("\n[가능한 원인 및 해결방법]");
			System.out.println("  1. JSON 형식이 잘못되었음 (따옴표, 쉼표, 중괄호 확인)");
			System.out.println("  2. 요청 본문이 불완전하게 종료됨 (특히 따옴표나 괄호가 닫혀있지 않음)");
			System.out.println("  3. Member 클래스의 필드명과 JSON 키가 일치하지 않음");
			System.out.println("  4. Content-Type이 'application/json'이 아님");
			System.out.println("  5. 요청 본문이 비어있거나 Null");
			
			System.out.println("\n========== [JWTAuthenticationFilter] 상세 스택 트레이스 ==========");
			e.printStackTrace();
			System.out.println("=========================================================================");
			throw new AuthenticationException("요청 처리 중 오류가 발생했습니다: " + e.getMessage()) {};
		}
	}
	
	/**
	 * 바이트 배열을 16진수 문자열로 변환 (디버깅용)
	 */
	private String bytesToHex(byte[] bytes) {
		StringBuilder sb = new StringBuilder();
		for (byte b : bytes) {
			sb.append(String.format("%02x ", b));
		}
		return sb.toString();
	}
	
	@Override
	protected void successfulAuthentication(HttpServletRequest request, HttpServletResponse response, 
			FilterChain chain, Authentication authResult) throws IOException, ServletException {
		System.out.println("[JWTAuthenticationFilter] successfulAuthentication - Login successful!");
		SecurityUser user = (SecurityUser) authResult.getPrincipal();
		Member member = user.getMember();
		String userId = member.getUserId();
		
		// 브라우저/기기 정보 추출
		String userAgent = request.getHeader("User-Agent");
		if (userAgent == null) {
			userAgent = "Unknown";
		}
		
		// Remote IP 및 PORT 정보 추출
		String remoteAddr = getRemoteAddress(request);
		int remotePort = request.getRemotePort();
		String remoteInfo = remoteAddr + ":" + remotePort;
		
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
	}
	
	@Override
	protected void unsuccessfulAuthentication(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException failed) throws IOException, ServletException {
		response.setStatus(HttpStatus.UNAUTHORIZED.value());
		response.setContentType(MediaType.APPLICATION_JSON_VALUE);
		response.setCharacterEncoding("UTF-8");
		
		responseDTO res = responseDTO.builder()
			.success(false)
			.errorMsg("로그인에 실패했습니다. 아이디 또는 비밀번호를 확인하세요.")
			.build();
		
		String responseBody = new ObjectMapper().writeValueAsString(res);
		response.getWriter().write(responseBody);
		response.getWriter().flush();
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
