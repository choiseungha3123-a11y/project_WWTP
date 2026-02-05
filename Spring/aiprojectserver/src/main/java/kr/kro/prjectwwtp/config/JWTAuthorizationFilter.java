package kr.kro.prjectwwtp.config;

import java.io.IOException;
import java.util.Optional;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthorizationFilter extends OncePerRequestFilter {
	private final MemberRepository memberRepo;
	private final TokenBlacklistManager tokenBlacklistManager;

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		JWTUtil.setMemberRepository(memberRepo);
		JWTUtil.setTokenBlacklistManager(tokenBlacklistManager);
		
		String requestPath = request.getRequestURI();
		String method = request.getMethod();
		String remoteAddr = Util.getRemoteAddress(request);
		System.out.println("\n========== [JWTAuthorizationFilter] START ==========");
		System.out.println("[JWTAuthorizationFilter] Method: " + method);
		System.out.println("[JWTAuthorizationFilter] Path: " + requestPath);
		System.out.println("[JWTAuthorizationFilter] IP: " + remoteAddr);
		
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
		
		// 토큰이 블랙리스트에 있는지 확인 (다른 기기에서 재로그인 시 이전 토큰 무효화)
		if (tokenBlacklistManager.isTokenBlacklisted(jwtToken)) {
			System.out.println("[JWTAuthorizationFilter] ⚠️  Token is blacklisted (invalidated by new login from another browser/device)");
			System.out.println("========== [JWTAuthorizationFilter] END (BLACKLISTED TOKEN) ==========\n");
			response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
			response.setContentType(MediaType.APPLICATION_JSON_VALUE);
			response.setCharacterEncoding("UTF-8");
			responseDTO errorRes = responseDTO.builder()
				.success(false)
				.errorMsg("다른 기기/브라우저에서 로그인되어 현재 로그인이 만료되었습니다. 다시 로그인해주세요.")
				.build();
			String errorJson = new ObjectMapper().writeValueAsString(errorRes);
			response.getWriter().write(errorJson);
			response.getWriter().flush();
			return;
		}
		
		try {
			// 토큰에서 userid 추출
			String userid = JWTUtil.getClaim(jwtToken, JWTUtil.useridClaim);
			System.out.println("[JWTAuthorizationFilter] Extracted userid: " + userid);
			
			if (userid == null) {
				System.out.println("[JWTAuthorizationFilter] userid is null, passing to next filter");
				System.out.println("========== [JWTAuthorizationFilter] END (NO USERID) ==========\n");
				filterChain.doFilter(request, response);
				return;
			}
			
			// DB에서 사용자 조회
			Optional<Member> opt = memberRepo.findByUserId(userid);
			if(!opt.isPresent()) {
				System.out.println("[JWTAuthorizationFilter] User not found: " + userid);
				System.out.println("========== [JWTAuthorizationFilter] END (USER NOT FOUND) ==========\n");
				filterChain.doFilter(request, response);
				return;
			}
			
			Member member = opt.get();
			System.out.println("[JWTAuthorizationFilter] Found member: " + member.getUserId());
			
			// SecurityUser 객체 생성
			SecurityUser user = new SecurityUser(member);
			
			// 인증 객체 생성 및 SecurityContext에 등록
			Authentication auth = new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities());
			SecurityContextHolder.getContext().setAuthentication(auth);
			System.out.println("[JWTAuthorizationFilter] Authentication set for: " + userid);
			System.out.println("========== [JWTAuthorizationFilter] END (SUCCESS) ==========\n");
		} catch (Exception e) {
			System.out.println("[JWTAuthorizationFilter] Error during token validation: " + e.getMessage());
			System.out.println("========== [JWTAuthorizationFilter] END (ERROR) ==========\n");
			//e.printStackTrace();
		}
		
		// SecurityFilterChain의 다음 필터로 이동
		filterChain.doFilter(request, response);
	}
	
	// ...existing code...
}