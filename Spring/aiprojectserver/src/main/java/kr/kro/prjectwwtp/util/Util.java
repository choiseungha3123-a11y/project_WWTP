package kr.kro.prjectwwtp.util;

import jakarta.servlet.http.HttpServletRequest;

public class Util {
	
	/**
	 * 클라이언트의 실제 IP 주소 추출
	 * 프록시 환경에서도 올바른 IP를 가져오도록 처리
	 */
	public static String getRemoteAddress(HttpServletRequest request) {
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
