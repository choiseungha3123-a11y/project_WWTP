package kr.kro.prjectwwtp.service;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.AccessLog;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.persistence.AccessLogRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class AccessLogService {
	private final AccessLogRepository logRepo;
	
	public void addLog(Member member, String userAgent, String remoteInfo, String method, String requestURI, String errorMsg) {
		Member logMember = null;
		if(member!= null && member.getUserNo() != 0)
			logMember = member;
		// 로그 추가
		logRepo.save(AccessLog.builder()
							.member(logMember)
							.userAgent(userAgent)
							.remoteInfo(remoteInfo)
							.method(method)
							.requestURI(requestURI)
							.errorMsg(errorMsg)
							.build());
	}
}
