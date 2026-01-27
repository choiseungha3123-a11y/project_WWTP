package kr.kro.prjectwwtp.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.session.SessionInformation;
import org.springframework.security.core.session.SessionRegistry;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

@Service
public class SessionService {
	@Autowired
    private SessionRegistry sessionRegistry;

    /**
     * 특정 사용자 아이디로 모든 세션을 찾아 만료시킴 (동시 로그인 차단 해제 등)
     */
    public void expireUserSessions(String username) {
    	System.out.println("expireUserSessions : " + username);
        // 1. 현재 접속 중인 모든 사용자(Principals)를 가져옴
        List<Object> allPrincipals = sessionRegistry.getAllPrincipals();

        for (Object principal : allPrincipals) {
            if (principal instanceof UserDetails) {
                UserDetails user = (UserDetails) principal;
                
                // 찾는 아이디와 일치하면
                if (user.getUsername().equals(username)) {
                    // 2. 해당 유저의 모든 세션 정보 호출 (false: 만료된 세션 제외)
                    List<SessionInformation> sessions = sessionRegistry.getAllSessions(principal, false);
                    
                    for (SessionInformation session : sessions) {
                        session.expireNow(); // 즉시 세션 만료 처리
                    }
                }
            }
        }
    }
}
