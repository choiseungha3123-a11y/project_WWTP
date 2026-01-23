package kr.kro.prjectwwtp.config;

import java.util.Arrays;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class SecurityConfig {
	private final AuthenticationConfiguration authenticationConfiguration;
	private final AuthenticationSuccessHandler oauth2SuccessHandler;
	
	@Bean
	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		http.cors(cors->cors.configurationSource(corsSource()));
		// CSRF 보호 비활성화 (CsrfFilter 제거)
		http.csrf(csrf->csrf.disable());
		http.httpBasic(basic->basic.disable());
		
		// 접근권한 설정
		http.authorizeHttpRequests(auth->auth
				.requestMatchers("/member/**").authenticated()
				.requestMatchers("/admin/**").hasRole("ADMIN")
				.anyRequest().permitAll());	// AuthorizationFilter 등록
	
		http.sessionManagement(sm ->sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS));
		http.addFilter(new JWTAuthenticationFilter(authenticationConfiguration.getAuthenticationManager()));
		//http.addFilterBefore(new JWTAuthorizationFilter(memberRepo), AuthorizationFilter.class);
		
		//http.formLogin(form->form.loginPage("/system/login").defaultSuccessUrl("/member/main", true));
		http.exceptionHandling(ex->ex.accessDeniedPage("/system/accessDenied"));
		http.logout(logout->logout
				.logoutUrl("/system/logout")	// 로그아웃 요청 주소
				.logoutSuccessUrl("/")			// 로그 아웃 성공 후 이동할 주소
				.invalidateHttpSession(true)	// 세션 삭제
				.clearAuthentication(true)		// 인증 정보 삭제
				.deleteCookies("JSESSIONID")	// 쿠키 삭제
				);
		
		
		return http.build();
	}

	private CorsConfigurationSource corsSource() {
		CorsConfiguration config = new CorsConfiguration();
		config.setAllowedOriginPatterns(
			Arrays.asList(
					"http://localhost:3000", 
					"http://127.0.0.1:3000"));
		config.addAllowedMethod(CorsConfiguration.ALL);		// 요청을 허용할 Method
		config.addAllowedHeader(CorsConfiguration.ALL);		// 요청을 허용할 Header
		config.setAllowCredentials(true);					// 요청/응답에 자격증명정보/쿠키 포함을 허용 여부
		config.addExposedHeader(HttpHeaders.AUTHORIZATION);	// 응답 Header에 Authorization 추가
		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
		source.registerCorsConfiguration("/**", config);
		return source;
	}
}
