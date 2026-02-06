package kr.kro.prjectwwtp.config;

import java.util.Arrays;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
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
import kr.kro.prjectwwtp.service.AccessLogService;
import kr.kro.prjectwwtp.service.LoginLogService;
import kr.kro.prjectwwtp.service.SessionService;
import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
@EnableWebSecurity
public class SecurityConfig {

	private final MemberRepository memberRepo;
	private final AuthenticationSuccessHandler oauth2SuccessHandler;
	private final AuthenticationFailureHandler oauth2FailurHandler;
	private final TokenBlacklistManager tokenBlacklistManager;
	private final SessionService sessionService;
	private final AccessLogService logService;
	private final LoginLogService loginService;
	
	@Bean
	AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
		return authenticationConfiguration.getAuthenticationManager();
	}
	
	@Bean
	SecurityFilterChain securityFilterChain(HttpSecurity http, AuthenticationManager authenticationManager) throws Exception {
		// JWT 인증 필터 생성 (로그인 처리)
		JWTAuthenticationFilter jwtAuthenticationFilter = new JWTAuthenticationFilter(authenticationManager, tokenBlacklistManager, sessionService, logService, loginService);
		// 로그인 엔드포인트 지정
		jwtAuthenticationFilter.setFilterProcessesUrl("/api/member/login");
		
		// CORS 설정
		http.cors(cors -> cors.configurationSource(corsConfigurationSource()));
		
		// CSRF 비활성화
		http.csrf(csrf -> csrf.disable());
		
		// HTTP Basic 인증 비활성화
		http.httpBasic(basic -> basic.disable());
		
		// 세션 정책: STATELESS (JWT 사용)
		http.sessionManagement(session -> session
			.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
			.maximumSessions(1)
			.maxSessionsPreventsLogin(false)
			.sessionRegistry(sessionRegistry())
			.expiredUrl("/login?expired=true"));
		
		// 접근 권한 설정
		http.authorizeHttpRequests(auth -> auth
			// 공개 접근 가능 (필터 적용 안 함)
			.requestMatchers("/v3/api-docs/**").permitAll()
			.requestMatchers("/swagger-ui/**").permitAll()
			.requestMatchers("/swagger-ui.html").permitAll()
			.requestMatchers("/swagger-resources/**").permitAll()
			.requestMatchers("/static/**").permitAll()
			
			.requestMatchers("/api/member/login").permitAll()
			.requestMatchers("/api/member/checkId").permitAll()
			.requestMatchers("/api/member/validateKey").permitAll()
			.requestMatchers("/api/weather/list").permitAll()
			.requestMatchers("/api/tmsOrigin/tmsList").permitAll()
			
			// 인증 필요 - 이 경로들은 JWT 필터를 통과해야 함
			.requestMatchers("/api/member/logout").authenticated()
			.requestMatchers("/api/member/modify").authenticated()
			.requestMatchers("/api/member/delete").authenticated()

			// 관리자 권한 필요
			.requestMatchers("/api/member/list").hasRole("ADMIN")
			.requestMatchers("/api/member/create").hasRole("ADMIN")
			.requestMatchers("/api/member/validateEmail").hasRole("ADMIN")
			.requestMatchers("/api/memo/**").hasAnyRole("MEMBER", "ADMIN")
			.requestMatchers("/api/tmsOrigin/**").hasRole("ADMIN")
			.requestMatchers("/api/weather/modify").hasRole("ADMIN")
			.requestMatchers("/api/admin/**").hasRole("ADMIN")
			.requestMatchers("/admin/**").hasRole("ADMIN")
			// 그 외는 허용
			.anyRequest().permitAll());
		
		// JWT 인증 필터 추가 (로그인 처리)
		http.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
		
		// JWT 인가 필터 추가 (토큰 검증용) - JWTAuthenticationFilter 이후에 실행
		JWTAuthorizationFilter jwtAuthorizationFilter = new JWTAuthorizationFilter(memberRepo, tokenBlacklistManager);
		http.addFilterAfter(jwtAuthorizationFilter, UsernamePasswordAuthenticationFilter.class);
		
		// 폼 로그인 설정 비활성화
		http.formLogin(form -> form.disable());
		
		// OAuth2 인증 추가
		http.oauth2Login(oauth2->oauth2
				.authorizationEndpoint(endpoint -> endpoint.baseUri("/api/oauth2/authorization"))
				.redirectionEndpoint(endpoint -> endpoint.baseUri("/api/oauth2/code/*"))
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
