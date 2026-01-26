package kr.kro.prjectwwtp.config;

import java.io.IOException;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;

import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.util.JWTUtil;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JWTAuthenticationFilter extends UsernamePasswordAuthenticationFilter{
	private final AuthenticationManager authenticatonManager;	// 인증을 실행하는 객체(ProviderManager)
	
	// POST/login 요청이 왔을 때 인증을 시도하는 메소드
	@Override
	public Authentication attemptAuthentication(HttpServletRequest request, HttpServletResponse response)
			throws AuthenticationException {
		// TODO Auto-generated method stub
		ObjectMapper mapper = new ObjectMapper();								// JSON/Object Mapping 객체 생성
		
		Member member = null;
		try {
			member = mapper.readValue(request.getInputStream(), Member.class);	// request에서 [username/password]를 읽어서 Member 객체를 생성
		} catch (Exception e) {
			// TODO: handle exception
			return null;														// 예외가 발생하면 null 리턴 -> unsuccessfulAuthentication 호출
		}
		if(member == null)
			return null;														// member가 null이면 null 리턴 -> unsuccessfulAuthentication 호출
		
		// Security에게 자격 증명 요청에 필요한 객체 생성
		Authentication authToken = new UsernamePasswordAuthenticationToken(member.getUserId(), member.getPassword());
		
		// 인증 메서드 호출 -> UserDetailsService의 loadUserByUsername에서 DB로부터 사용자 정보를 읽어온 뒤 사용자 입력 정보(member)와 비교 검증
		// 자격 증명에 성공하면 Authentication객체를 만들어서 리턴하면 successfulAuthenticatin 호출, 실패하면 unsuccessfulAuthentication 호출
		return authenticatonManager.authenticate(authToken);
	}
	
	// 인증이 성공했을 때 실행되는 후처리 메소드
	@Override
	protected void successfulAuthentication(HttpServletRequest request, HttpServletResponse response, FilterChain chain,
			Authentication authResult) throws IOException, ServletException {
		// TODO Auto-generated method stub
		SecurityUser user = (SecurityUser)authResult.getPrincipal();					// loadUserByUsername에서 만든 객체가 authResult에 담겨져 있다.
		//System.out.println("[JWTAuthenticationFilter]auth:" + user);	// user 객체를 콘솔에 출력해서 확인
		String token = JWTUtil.getJWT(user.getMember());				// username으로 JWT 생성. 이것은 예시로서 필요에 따라 정보 추가
		response.addHeader(HttpHeaders.AUTHORIZATION, token);			// Response Header[Authorization]에 JWT를 저장해서 응답한다.
		response.setStatus(HttpStatus.OK.value());						// 자격 증명 성공 응답코드 리턴
	}
	
	// 인증에 실패했을 때 실행되는 후처리 메소드
	@Override
	protected void unsuccessfulAuthentication(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException failed) throws IOException, ServletException {
		// TODO Auto-generated method stub
		//System.out.println("unsuccessfulAuthentication:" + failed);		// 콘솔에 출력해서 확인
		response.setStatus(HttpStatus.UNAUTHORIZED.value());			// 자격 증명 실패 응답코드 리턴
	}
}
