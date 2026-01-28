package kr.kro.prjectwwtp.config;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Component;

/**
 * JWT 토큰 블랙리스트 관리
 * 여러 브라우저/기기에서의 중복 로그인 방지
 * 사용자당 하나의 활성 토큰만 유지
 */
@Component
public class TokenBlacklistManager {
	
	// userId -> 현재 활성 토큰 맵핑 (사용자당 1개만 유지)
	private final Map<String, String> activeTokens = new ConcurrentHashMap<>();
	
	// 무효화된 토큰 저장소 (토큰 -> 무효화 시간)
	private final Map<String, Long> blacklistedTokens = new ConcurrentHashMap<>();
	
	// 사용자별 마지막 로그인 정보
	private final Map<String, LoginInfo> loginInfoMap = new ConcurrentHashMap<>();
	
	/**
	 * 로그인 정보 클래스
	 */
	public static class LoginInfo {
		public String userId;
		public String token;
		public long loginTime;
		public String userAgent;
		public String remoteInfo;
		
		public LoginInfo(String userId, String token, String userAgent, String remoteInfo) {
			this.userId = userId;
			this.token = token;
			this.loginTime = System.currentTimeMillis();
			this.userAgent = userAgent;
			this.remoteInfo = remoteInfo;
		}
		
		@Override
		public String toString() {
			return String.format("[LoginInfo] userId=%s, token=%s, loginTime=%d, userAgent=%s, remoteInfo=%s", 
				userId, token.substring(0, Math.min(20, token.length())) + "...", loginTime, userAgent, remoteInfo);
		}
	}
	
	/**
	 * 새 토큰으로 업데이트하고 기존 토큰 무효화 (중복 로그인 방지)
	 * @param userId 사용자 ID
	 * @param newToken 새로 발급된 토큰
	 * @param userAgent 사용자 에이전트 (브라우저 정보)
	 * @param remoteInfo Remote IP:PORT 정보
	 */
	public void registerNewToken(String userId, String newToken, String userAgent, String remoteInfo) {
		System.out.println("[TokenBlacklistManager] ====== New Login Attempt ======");
		System.out.println("[TokenBlacklistManager] userId: " + userId);
		System.out.println("[TokenBlacklistManager] userAgent: " + userAgent);
		System.out.println("[TokenBlacklistManager] remoteInfo: " + remoteInfo);
		
		// 기존 토큰이 있으면 블랙리스트에 추가 (다른 브라우저/기기의 로그인 무효화)
		String oldToken = activeTokens.get(userId);
		if (oldToken != null && !oldToken.equals(newToken)) {
			System.out.println("[TokenBlacklistManager] ⚠️  Previous login invalidated for user: " + userId);
			System.out.println("[TokenBlacklistManager] Old token blacklisted");
			blacklistedTokens.put(oldToken, System.currentTimeMillis());
		}
		
		// 새 토큰 등록 (현재 로그인만 활성 상태)
		activeTokens.put(userId, newToken);
		
		// 로그인 정보 저장
		LoginInfo newLoginInfo = new LoginInfo(userId, newToken, userAgent, remoteInfo);
		loginInfoMap.put(userId, newLoginInfo);
		
		System.out.println("[TokenBlacklistManager] ✅ New token registered for: " + userId);
		System.out.println("[TokenBlacklistManager] " + newLoginInfo);
		System.out.println("[TokenBlacklistManager] ============================\n");
	}
	
	/**
	 * 토큰이 블랙리스트에 있는지 확인
	 * @param token 확인할 토큰
	 * @return 블랙리스트 여부
	 */
	public boolean isTokenBlacklisted(String token) {
		boolean isBlacklisted = blacklistedTokens.containsKey(token);
		if (isBlacklisted) {
			System.out.println("[TokenBlacklistManager] ⚠️  Token is blacklisted (invalidated by new login from another browser/device)");
		}
		return isBlacklisted;
	}
	
	/**
	 * 사용자의 활성 토큰 확인
	 * @param userId 사용자 ID
	 * @return 활성 토큰 (없으면 null)
	 */
	public String getActiveToken(String userId) {
		return activeTokens.get(userId);
	}
	
	/**
	 * 토큰이 사용자의 현재 활성 토큰인지 확인
	 * @param userId 사용자 ID
	 * @param token 확인할 토큰
	 * @return 활성 토큰 여부
	 */
	public boolean isActiveToken(String userId, String token) {
		String activeToken = activeTokens.get(userId);
		return activeToken != null && activeToken.equals(token);
	}
	
	/**
	 * 로그아웃 시 토큰 무효화
	 * @param userId 사용자 ID
	 */
	public void invalidateToken(String userId) {
		System.out.println("[TokenBlacklistManager] Invalidating token for user: " + userId);
		String token = activeTokens.remove(userId);
		if (token != null) {
			blacklistedTokens.put(token, System.currentTimeMillis());
			loginInfoMap.remove(userId);
			System.out.println("[TokenBlacklistManager] ✅ Token invalidated and logout completed");
		}
	}
	
	/**
	 * 사용자의 로그인 정보 조회
	 * @param userId 사용자 ID
	 * @return 로그인 정보
	 */
	public LoginInfo getLoginInfo(String userId) {
		return loginInfoMap.get(userId);
	}
	
	/**
	 * 오래된 블랙리스트 항목 정리
	 * 토큰 만료 시간(1시간)보다 오래된 항목은 제거
	 */
	public void cleanupExpiredBlacklist() {
		long expirationTime = 60 * 60 * 1000; // 1시간
		long currentTime = System.currentTimeMillis();
		
		int removedCount = (int) blacklistedTokens.entrySet().stream()
			.filter(entry -> (currentTime - entry.getValue()) > expirationTime)
			.count();
		
		blacklistedTokens.entrySet().removeIf(entry -> 
			(currentTime - entry.getValue()) > expirationTime
		);
		
		if (removedCount > 0) {
			System.out.println("[TokenBlacklistManager] Cleanup: " + removedCount + " expired tokens removed");
		}
	}
}

