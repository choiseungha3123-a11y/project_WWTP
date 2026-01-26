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
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthorizationFilter extends OncePerRequestFilter {
	private final MemberRepository memberRepo;

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		// TODO Auto-generated method stub
		String jwtToken = request.getHeader(HttpHeaders.AUTHORIZATION);
		if(jwtToken == null ||!jwtToken.startsWith(JWTUtil.useridClaim)) {
			filterChain.doFilter(request, response);
			return;
		}
		// 토큰에서 username 추출
		String userid = JWTUtil.getClaim(jwtToken, JWTUtil.useridClaim);
		SecurityUser user = null;
		Member member = null;
		if (userid != null) {
			Optional<Member> opt = memberRepo.findByUserId(userid);
			if(!opt.isPresent()) {
				//System.out.println("[JWTAuthorizationFilter]not found user!");
				filterChain.doFilter(request, response);
				return;
			}
			member = opt.get();
			//System.out.println("[JWTAuthorizationFilter]" + member);
		} 
		// DB에서 읽은 사용자 정보를 이용해서 UserDetails 타입의 객체를 만들어서
		user = new SecurityUser(member);
		// 인증 객체 생성 : 사용자명과 권한 관리를 위한 정보를 입력(암호는 필요 없음)
		Authentication auth = new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities());
		// SecurityContext에 등록
		SecurityContextHolder.getContext().setAuthentication(auth);
		// SecurityFilterChain의 다음 필터로 이동
		filterChain.doFilter(request, response);
	}
	
	@Override
	protected boolean shouldNotFilter(HttpServletRequest request) throws ServletException {
		// TODO Auto-generated method stub
		//return super.shouldNotFilter(request);
		String path= request.getRequestURI();
		return path.startsWith("/ve/api-docs") || path.startsWith("/swagger-ui");
	}
}
