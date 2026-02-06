package kr.kro.prjectwwtp.util;

import java.util.Date;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.Claim;

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
	
	private static String API_KEY;
	private static final String UserNoClaim = "UserNo";
	private static final long MSEC = 10 * 60 * 1000;	// 10분
	
	public static void setKey(String key) {
		API_KEY = key;
	}
	
	public static String getTempKey(Long userNo) {
		String key = JWT.create()
				.withClaim(UserNoClaim, userNo.toString())
				.withExpiresAt(new Date(System.currentTimeMillis()+MSEC))
				.sign(Algorithm.HMAC256(API_KEY));
		return key;
	}
	
	public static boolean isExpired(String tempKey)
	{
		boolean result = true;
		try {
			result = JWT.require(Algorithm.HMAC256(API_KEY)).build()
					.verify(tempKey).getExpiresAt().before(new Date());
		}
		catch(Exception e)
		{
			System.out.println("토큰 만료");
			result = false;
		}
		return result;
	}
	
	public static String getClaim(String token, String cname) {
		Claim claim = JWT.require(Algorithm.HMAC256(API_KEY)).build()
						.verify(token).getClaim(cname);
		if (claim.isMissing() || claim.isNull()) return null;
		return claim.asString();
	}
		
	public static Long pareKey(String tempKey) {
		Long userNo = -1L;
		try {
			userNo = Long.parseLong(getClaim(tempKey, UserNoClaim));
		}
		catch(Exception e)
		{
			System.out.println("토큰 만료");
			userNo = -1L;
		}
		return userNo;
	}

}
