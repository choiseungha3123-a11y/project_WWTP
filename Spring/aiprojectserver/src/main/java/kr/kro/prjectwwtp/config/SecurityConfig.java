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
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.session.HttpSessionEventPublisher;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.service.AccessLogService;
import kr.kro.prjectwwtp.service.SessionService;
import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
@EnableWebSecurity
public class SecurityConfig {

	private final MemberRepository memberRepo;
	private final AuthenticationSuccessHandler oauth2SuccessHandler;
	private final TokenBlacklistManager tokenBlacklistManager;
	private final SessionService sessionService;
	private final AccessLogService logService;
	
	@Bean
	public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
		return authenticationConfiguration.getAuthenticationManager();
	}
	
	@Bean
	public SecurityFilterChain securityFilterChain(HttpSecurity http, AuthenticationManager authenticationManager) throws Exception {
		// JWT 인증 필터 생성 (로그인 처리)
		JWTAuthenticationFilter jwtAuthenticationFilter = new JWTAuthenticationFilter(authenticationManager, tokenBlacklistManager, sessionService, logService);
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
			.requestMatchers("/api/weather/list").permitAll()
			
			// 인증 필요 - 이 경로들은 JWT 필터를 통과해야 함
			.requestMatchers("/api/member/logout").authenticated()
			.requestMatchers("/api/member/modifyMember").authenticated()
			.requestMatchers("/api/member/deleteMember").authenticated()
			.requestMatchers("/api/memo/**").authenticated()

			// 관리자 권한 필요
			.requestMatchers("/api/member/addMember").hasRole("ADMIN")
			.requestMatchers("/api/member/listMember").hasRole("ADMIN")
			.requestMatchers("/api/weather/modify").hasRole("ADMIN")
			.requestMatchers("/admin/**").hasRole("ADMIN")
			// 그 외는 허용
			.anyRequest().permitAll());
		
		// JWT 인증 필터 추가 (로그인 처리)
		System.out.println("[SecurityConfig] Adding JWTAuthenticationFilter at position: BEFORE UsernamePasswordAuthenticationFilter");
		http.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
		
		// JWT 인가 필터 추가 (토큰 검증용) - JWTAuthenticationFilter 이후에 실행
		System.out.println("[SecurityConfig] Adding JWTAuthorizationFilter at position: AFTER UsernamePasswordAuthenticationFilter");
		JWTAuthorizationFilter jwtAuthorizationFilter = new JWTAuthorizationFilter(memberRepo, tokenBlacklistManager);
		http.addFilterAfter(jwtAuthorizationFilter, UsernamePasswordAuthenticationFilter.class);
		System.out.println("[SecurityConfig] Filter registration complete!");
		
		// 폼 로그인 설정 비활성화
		http.formLogin(form -> form.disable());
		
		// OAuth2 인증 추가
		http.oauth2Login(oauth2->oauth2.successHandler(oauth2SuccessHandler));
		System.out.println("[SecurityConfig] Filter oauth2Login complete!");
		
		// 예외 처리
		http.exceptionHandling(ex -> ex.accessDeniedPage("/system/accessDenied"));
		
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
	public HttpSessionEventPublisher httpSessionEventPublisher() {
		return new HttpSessionEventPublisher();
	}
	
	@Bean
	public SessionRegistry sessionRegistry() {
		return new SessionRegistryImpl();
	}
	
	@Value("${spring.AllowedOriginPatterns}")
	private String[] patterns;
	
	@Bean
	public CorsConfigurationSource corsConfigurationSource() {
		System.out.println(Arrays.toString(patterns));
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
