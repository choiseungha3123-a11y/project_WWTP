package kr.kro.prjectwwtp.config;

import java.io.IOException;
import java.util.Optional;

import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.service.LogService;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthorizationFilter extends OncePerRequestFilter {
	private final MemberRepository memberRepo;
	private final LogService logService;

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		Member member = null;
		String method = request.getMethod();
		String userAgent = request.getHeader("User-Agent");
		if (userAgent == null) {
			userAgent = "Unknown";
		}
		String remoteAddr = Util.getRemoteAddress(request);
		int remotePort = request.getRemotePort();
		String remoteInfo = remoteAddr + ":" + remotePort;
		String errorMsg = null;
		
		JWTUtil.setMemberRepository(memberRepo);
		
		String requestPath = request.getRequestURI();
		System.out.println("\n========== [JWTAuthorizationFilter] START ==========");
		System.out.println("[JWTAuthorizationFilter] Method: " + method);
		System.out.println("[JWTAuthorizationFilter] Path: " + requestPath);
		System.out.println("[JWTAuthorizationFilter] IP: " + remoteInfo);
		
		String jwtToken = request.getHeader(HttpHeaders.AUTHORIZATION);
		System.out.println("[JWTAuthorizationFilter] Authorization header: " + (jwtToken != null ? "존재함 (" + jwtToken.substring(0, Math.min(20, jwtToken.length())) + "...)" : "없음"));
		System.out.println("[JWTAuthorizationFilter] Prefix check: " + (jwtToken != null ? jwtToken.startsWith(JWTUtil.prefix) : "null"));
		
		// 토큰이 없거나 "Bearer " 프리픽스가 없으면 필터 패스
		if(jwtToken == null || !jwtToken.startsWith(JWTUtil.prefix)) {
			System.out.println("[JWTAuthorizationFilter] No valid token, passing to next filter");
			System.out.println("========== [JWTAuthorizationFilter] END (NO TOKEN) ==========\n");
			filterChain.doFilter(request, response);
			return;
		}
		
		try {
			// 토큰에서 userid 추출
			String userid = JWTUtil.getClaim(jwtToken, JWTUtil.useridClaim);
			System.out.println("[JWTAuthorizationFilter] Extracted userid: " + userid);
			
			if (userid == null) {
				System.out.println("[JWTAuthorizationFilter] userid is null, passing to next filter");
				filterChain.doFilter(request, response);
				return;
			}
			
			// DB에서 사용자 조회
			Optional<Member> opt = memberRepo.findByUserId(userid);
			if(!opt.isPresent()) {
				System.out.println("[JWTAuthorizationFilter] User not found: " + userid);
				filterChain.doFilter(request, response);
				return;
			}
			
			member = opt.get();
			System.out.println("[JWTAuthorizationFilter] Found member: " + member.getUserId());
			
			// SecurityUser 객체 생성
			SecurityUser user = new SecurityUser(member);
			
			// 인증 객체 생성 및 SecurityContext에 등록
			Authentication auth = new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities());
			SecurityContextHolder.getContext().setAuthentication(auth);
			System.out.println("[JWTAuthorizationFilter] Authentication set for: " + userid);
			System.out.println("========== [JWTAuthorizationFilter] END (SUCCESS) ==========\n");
		} catch (Exception e) {
			errorMsg = e.getMessage();
			System.out.println("[JWTAuthorizationFilter] Error during token validation: " + errorMsg);
			System.out.println("========== [JWTAuthorizationFilter] END (ERROR) ==========\n");
			if(!errorMsg.startsWith("The Token has expired"))
				logService.addErrorLog("JWTAuthorizationFilter.java", "doFilterInternal()", errorMsg);
			//e.printStackTrace();
		}finally {
			if(!requestPath.endsWith("/health")
					&& !requestPath.endsWith("/boardView")
					&&  (errorMsg != null && !errorMsg.startsWith("The Token has expired")))
				logService.addAccessLog(member, userAgent, remoteInfo, method, requestPath, errorMsg);
		}
		
		// SecurityFilterChain의 다음 필터로 이동
		filterChain.doFilter(request, response);
	}
	
	// ...existing code...
}