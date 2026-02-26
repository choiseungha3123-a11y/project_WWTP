package kr.kro.prjectwwtp.config;

import java.util.Arrays;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.session.SessionRegistry;
import org.springframework.security.core.session.SessionRegistryImpl;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.session.HttpSessionEventPublisher;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import jakarta.servlet.http.HttpServletResponse;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.service.LogService;
import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class SecurityConfig {
	private final AuthenticationSuccessHandler oauth2SuccessHandler;
	private final AuthenticationFailureHandler oauth2FailurHandler;
	private final MemberRepository memberRepo;
	private final LogService logService;
	
	@Bean
	SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		// CORS 설정
		http.cors(cors -> cors.configurationSource(corsConfigurationSource()));
		
		// CSRF 비활성화
		http.csrf(csrf -> csrf.disable());
		
		// HTTP Basic 인증 비활성화
		http.httpBasic(basic -> basic.disable());
		
		// 세션 정책: STATELESS (JWT 사용)
		http.sessionManagement(session -> session
			.sessionCreationPolicy(SessionCreationPolicy.STATELESS));
		
		// 폼 로그인 설정 비활성화
		http.formLogin(form -> form.disable());
		
		// 접근 권한 설정
		http.authorizeHttpRequests(auth -> auth
			// 공개 접근 가능 (필터 적용 안 함)
			.requestMatchers("/api/v3/api-docs/**").permitAll()
			.requestMatchers("/api/swagger-ui/**").permitAll()
			.requestMatchers("/api/swagger-ui.html").permitAll()
			.requestMatchers("/api/swagger-resources/**").permitAll()
			.requestMatchers("/api/static/**").permitAll()
			
			// 대시보드 컨트롤
			.requestMatchers("/api/board/health").permitAll()
			.requestMatchers("/api/board/memo/image").permitAll()
			.requestMatchers("/api/board/makeFakeNow").permitAll()
			.requestMatchers("/api/board/**").hasAnyRole("MEMBER", "ADMIN")
			
			// 유입유량(AWS) 관련
			.requestMatchers("/api/flowOrigin/upload").hasRole("ADMIN")
			.requestMatchers("/api/flowOrigin/list").hasRole("ADMIN")
			//.requestMatchers("/api/flowOrigin/flowList").hasAnyRole("MEMBER", "ADMIN")
			.requestMatchers("/api/flowOrigin/flowList").permitAll()
			
			// TMS 관련
			.requestMatchers("/api/tmsOrigin/upload").hasRole("ADMIN")
			.requestMatchers("/api/tmsOrigin/list").hasRole("ADMIN")
			//.requestMatchers("/api/tmsOrigin/tmsList").hasAnyRole("MEMBER", "ADMIN")
			.requestMatchers("/api/tmsOrigin/flowList").permitAll()
			
			// 메일 관련
			.requestMatchers("/api/mail/sendTo").hasRole("ADMIN")
			
			// 맴버 관련
			.requestMatchers("/api/member/login").permitAll()
			.requestMatchers("/api/member/logout").authenticated()
			.requestMatchers("/api/member/list").hasRole("ADMIN")
			.requestMatchers("/api/member/checkId").permitAll()
			.requestMatchers("/api/member/checkEmail").permitAll()
			.requestMatchers("/api/member/validateEmail").hasRole("ADMIN")
			.requestMatchers("/api/member/validateKey").permitAll()
			.requestMatchers("/api/member/create").hasRole("ADMIN")
			.requestMatchers("/api/member/modify").authenticated()
			.requestMatchers("/api/member/delete").authenticated()
			
			// OAuth2 관련
			.requestMatchers("/api/oauth2/**").permitAll()
			
			// 날씨 수집 관련
			.requestMatchers("/api/weather/list").permitAll()
			.requestMatchers("/api/weather/modify").hasRole("ADMIN")
			
			.requestMatchers("/api/public/**").permitAll()

			// 그 외는 허용
			.anyRequest().permitAll()
		);
		
		// JWT 인가 필터 추가 (토큰 검증용) - JWTAuthenticationFilter 이후에 실행
		JWTAuthorizationFilter jwtAuthorizationFilter = new JWTAuthorizationFilter(memberRepo, logService);
		http.addFilterAfter(jwtAuthorizationFilter, UsernamePasswordAuthenticationFilter.class);
		
		// OAuth2 인증 추가
		http.oauth2Login(oauth2->oauth2
				.authorizationEndpoint(endpoint -> endpoint.baseUri("/api/oauth2/authorization"))
				.redirectionEndpoint(endpoint -> endpoint.baseUri("/api/login/oauth2/code/*"))
				.failureHandler(oauth2FailurHandler)
				.successHandler(oauth2SuccessHandler));
		
		// 예외 처리
		http.exceptionHandling(conf -> conf
                // 인증되지 않은 사용자가 보호된 리소스에 접근할 때 (401)
                .authenticationEntryPoint((request, response, authException) -> {
                    response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                    response.setContentType("application/json;charset=UTF-8");
                    response.getWriter().write("{\"success\":false, \"dataSize\":0,\"dataList\":null,\"errorMsg\": \"로그인이 필요합니다.\"}");
                })
                // 인증은 되었으나 권한이 부족할 때 (403)
                .accessDeniedHandler((request, response, accessDeniedException) -> {
                    response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                    response.setContentType("application/json;charset=UTF-8");
                    response.getWriter().write("{\"success\":false, \"dataSize\":0,\"dataList\":null,\"errorMsg\": \"접근 권한이 없습니다.\"}");
                })
        );
		
		// 로그아웃 설정
		http.logout(logout -> logout
			.logoutUrl("/system/logout")
			.logoutSuccessUrl("/")
			.invalidateHttpSession(true)
			.clearAuthentication(true)
			.deleteCookies("JSESSIONID"));
		
		return http.build();
	}
	
	@Bean
	HttpSessionEventPublisher httpSessionEventPublisher() {
		return new HttpSessionEventPublisher();
	}
	
	@Bean
	SessionRegistry sessionRegistry() {
		return new SessionRegistryImpl();
	}
	
	@Value("${spring.AllowedOriginPatterns}")
	private String[] patterns;
	
	@Bean
	CorsConfigurationSource corsConfigurationSource() {
		System.out.println("[corsConfigurationSource] : " + Arrays.toString(patterns));
		CorsConfiguration config = new CorsConfiguration();
		config.setAllowedOriginPatterns(Arrays.asList(patterns));
		config.addAllowedMethod(CorsConfiguration.ALL);
		config.addAllowedHeader(CorsConfiguration.ALL);
		config.setAllowCredentials(true);
		config.addExposedHeader(HttpHeaders.AUTHORIZATION);
		
		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
		source.registerCorsConfiguration("/**", config);
		return source;
	}
}
