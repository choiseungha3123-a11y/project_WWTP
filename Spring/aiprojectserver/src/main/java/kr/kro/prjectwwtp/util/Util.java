package kr.kro.prjectwwtp.util;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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


	// parse helpers that treat empty string as null (explicit)
	public static Double parseDoubleOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null") || t.equalsIgnoreCase("-99.0") || t.equalsIgnoreCase("-99.9")) return null;
		return Double.parseDouble(t);
	}

	public static Integer parseIntOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null")) return null;
		return Integer.parseInt(t);
	}

	public static Long parseLongOrNull(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null")) return null;
		try {
			return Long.parseLong(t);
		} catch (NumberFormatException e) {
			return null;
		}
	}

	public static LocalDateTime parseDateTime(String s) {
		String str = s.trim();
		// try several common patterns
		String[] patterns = new String[] {
			"yyyy-MM-dd HH:mm:ss",
			"M/d/yy H:mm",
			"yyyy-MM-dd'T'HH:mm:ss",
			"yyyyMMddHHmm",
			"yyyyMMddHHmmss",
		};
		for (String p : patterns) {
			try {
				DateTimeFormatter f = DateTimeFormatter.ofPattern(p);
				return LocalDateTime.parse(str, f);
			} catch (Exception e) {
				// try next
			}
		}
		// try ISO parse
		try {
			return LocalDateTime.parse(str);
		} catch (Exception e) {
			throw new IllegalArgumentException("날짜 형식이 올바르지 않습니다: " + s);
		}
	}
	
	/**
	 * LocalDateTime을 문자열로 포맷
	 * 
	 * @param dateTime LocalDateTime 객체
	 * @return 포맷된 문자열 (yyyy-MM-dd HH:mm:ss)
	 */
	public static String formatDateTime(LocalDateTime dateTime) {
		if (dateTime == null) {
			return "";
		}
		return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
	}
	
	/**
	 * Double 값을 문자열로 포맷
	 * 
	 * @param value Double 값
	 * @return 포맷된 문자열 (null이면 빈 문자열)
	 */
	public static String formatDouble(Double value) {
		if (value == null || value.isNaN()) {
			return "";
		}
		return String.valueOf(value);
	}
	
	/**
	 * Integer 값을 문자열로 포맷
	 * 
	 * @param value Integer 값
	 * @return 포맷된 문자열 (null이면 빈 문자열)
	 */
	public static String formatInteger(Integer value) {
		if (value == null) {
			return "";
		}
		return String.valueOf(value);
	}
	
	/**
	 * 파일 경로를 정규화하여 절대 경로로 변환
	 * ~/Downloads/ 형태의 홈 디렉토리 경로도 처리
	 * 
	 * @param filePath 파일 경로 (상대/절대/홈 디렉토리 경로)
	 * @return 절대 경로 File 객체
	 */
	public static File resolveFilePath(String filePath) {
		File file = new File(filePath);
		
		// 이미 절대 경로인 경우
		if (file.isAbsolute()) {
			return file;
		}
		
		// 홈 디렉토리 경로 처리 (~/ 또는 ~\)
		if (filePath.startsWith("~" + File.separator) || filePath.startsWith("~/")) {
			String userHome = System.getProperty("user.home");
			String relativePath = filePath.substring(2); // ~/ 제거
			return new File(userHome, relativePath);
		}
		
		// 상대 경로인 경우 현재 작업 디렉토리 기준
		String workingDir = System.getProperty("user.dir");
		return new File(workingDir, filePath);
	}

}
