package kr.kro.prjectwwtp.service;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.security.core.session.SessionInformation;
import org.springframework.security.core.session.SessionRegistry;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

@Service
public class SessionService {
	@Autowired
	@Lazy  // 순환 참조 해결을 위해 지연 로딩
    private SessionRegistry sessionRegistry;
	
	// 사용자별 활성 세션 추적 (STATELESS 정책에서 수동 관리용)
	private final Map<String, SessionInfo> userSessions = new ConcurrentHashMap<>();

    /**
     * 특정 사용자 아이디로 모든 세션을 찾아 만료시킴 (동시 로그인 차단)
     */
    public void expireUserSessions(String username) {
    	System.out.println("[SessionService] expireUserSessions 호출: " + username);
    	
        // 1. SessionRegistry를 통해 활성 세션 찾기 (기존 로직)
        List<Object> allPrincipals = sessionRegistry.getAllPrincipals();
        System.out.println("[SessionService] 등록된 모든 Principal 수: " + allPrincipals.size());

        for (Object principal : allPrincipals) {
            if (principal instanceof UserDetails) {
                UserDetails user = (UserDetails) principal;
                
                // 찾는 아이디와 일치하면
                if (user.getUsername().equals(username)) {
                    System.out.println("[SessionService] Principal 매칭됨: " + user.getUsername());
                    // 2. 해당 유저의 모든 세션 정보 호출 (false: 만료된 세션 제외)
                    List<SessionInformation> sessions = sessionRegistry.getAllSessions(principal, false);
                    System.out.println("[SessionService] 만료할 세션 수: " + sessions.size());
                    
                    for (SessionInformation session : sessions) {
                        System.out.println("[SessionService] 세션 만료 처리: " + session.getSessionId());
                        session.expireNow(); // 즉시 세션 만료 처리
                    }
                }
            }
        }
        
        // 2. 수동으로 관리하는 사용자 세션도 확인 및 만료
        System.out.println("[SessionService] 수동 관리 세션 확인 - 현재 활성 사용자: " + userSessions.keySet());
        if (userSessions.containsKey(username)) {
            System.out.println("[SessionService] 수동 관리 세션 발견: " + username);
            SessionInfo sessionInfo = userSessions.remove(username);
            System.out.println("[SessionService] 이전 세션 제거됨 - Token: " + sessionInfo.getToken());
        }
    }
    
    /**
     * 새로운 로그인 세션을 등록 (JWT 토큰 기반)
     */
    public void registerNewSession(String username, String token, String userAgent, String remoteInfo) {
        System.out.println("[SessionService] 새 세션 등록: " + username);
        System.out.println("[SessionService] User Agent: " + userAgent);
        System.out.println("[SessionService] Remote Info: " + remoteInfo);
        
        SessionInfo sessionInfo = new SessionInfo(token, userAgent, remoteInfo, System.currentTimeMillis());
        userSessions.put(username, sessionInfo);
        
        System.out.println("[SessionService] 등록 완료 - 현재 활성 세션 수: " + userSessions.size());
    }
    
    /**
     * 사용자 세션 정보 조회
     */
    public SessionInfo getUserSession(String username) {
        return userSessions.get(username);
    }
    
    /**
     * 모든 활성 세션 조회 (디버깅용)
     */
    public Map<String, SessionInfo> getAllActiveSessions() {
        return new ConcurrentHashMap<>(userSessions);
    }
    
    /**
     * 세션 정보 클래스
     */
    public static class SessionInfo {
        private final String token;
        private final String userAgent;
        private final String remoteInfo;
        private final long loginTime;
        
        public SessionInfo(String token, String userAgent, String remoteInfo, long loginTime) {
            this.token = token;
            this.userAgent = userAgent;
            this.remoteInfo = remoteInfo;
            this.loginTime = loginTime;
        }
        
        public String getToken() { return token; }
        public String getUserAgent() { return userAgent; }
        public String getRemoteInfo() { return remoteInfo; }
        public long getLoginTime() { return loginTime; }
        
        @Override
        public String toString() {
            return "SessionInfo{" +
                    "token='" + token + '\'' +
                    ", userAgent='" + userAgent + '\'' +
                    ", remoteInfo='" + remoteInfo + '\'' +
                    ", loginTime=" + loginTime +
                    '}';
        }
    }
}
