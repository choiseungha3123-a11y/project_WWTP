package kr.kro.prjectwwtp.config;

import java.io.IOException;

import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.web.access.AccessDeniedHandler;
import org.springframework.stereotype.Component;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class CustomAccessDeniedHandler implements AccessDeniedHandler {
    @Override
    public void handle(HttpServletRequest request, HttpServletResponse response, 
                       AccessDeniedException accessDeniedException) throws IOException {
        // JSON 응답 직접 작성 또는 에러 페이지 리다이렉트
    	System.out.println("=============== AccessDeniedException 발생 ===============");
    	System.out.println("[요청 정보]");
    	System.out.println("  - 요청 URI: " + request.getRequestURI());
    	System.out.println("  - 요청 메서드: " + request.getMethod());
    	System.out.println("  - 요청자 IP: " + request.getRemoteAddr());
    	System.out.println("  - 사용자 Principal: " + request.getUserPrincipal());
    	
    	System.out.println("[예외 정보]");
    	System.out.println("  - 예외 메시지: " + accessDeniedException.getMessage());
    	System.out.println("  - 예외 타입: " + accessDeniedException.getClass().getName());
    	
    	System.out.println("[스택 트레이스]");
    	accessDeniedException.printStackTrace();
    	
    	System.out.println("=========================================================");
    	
    }
}