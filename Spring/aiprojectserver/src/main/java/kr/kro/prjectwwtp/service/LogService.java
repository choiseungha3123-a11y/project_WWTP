package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.AccessLog;
import kr.kro.prjectwwtp.domain.ErrorLog;
import kr.kro.prjectwwtp.domain.FlowLog;
import kr.kro.prjectwwtp.domain.LoginLog;
import kr.kro.prjectwwtp.domain.MailLog;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.MemoLog;
import kr.kro.prjectwwtp.domain.OutLierLog;
import kr.kro.prjectwwtp.domain.TmsLog;
import kr.kro.prjectwwtp.domain.WeatherApiLog;
import kr.kro.prjectwwtp.persistence.AccessLogRepository;
import kr.kro.prjectwwtp.persistence.ErrorLogRepository;
import kr.kro.prjectwwtp.persistence.FlowLogRepository;
import kr.kro.prjectwwtp.persistence.LoginLogRepository;
import kr.kro.prjectwwtp.persistence.MailLogRepository;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.persistence.MemoLogRepository;
import kr.kro.prjectwwtp.persistence.OutLierLogRepository;
import kr.kro.prjectwwtp.persistence.TmsLogRepository;
import kr.kro.prjectwwtp.persistence.WeatherAPILogRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class LogService {
	private final MemberRepository memberRepo;
	private final AccessLogRepository accessRepo;
	private final LoginLogRepository loginRepo;
	private final FlowLogRepository flowRepo;
	private final TmsLogRepository tmsRepo;
	private final MemoLogRepository memoRepo;
	private final WeatherAPILogRepository weatherRepo;
	private final OutLierLogRepository outLierRepo;
	private final MailLogRepository mailRepo;
	private final ErrorLogRepository errorRepo;
	
	public void addAccessLog(Member member, String userAgent, String remoteInfo, String method, String requestURI, String errorMsg) {
		Member logMember = null;
		if(member!= null && member.getUserNo() != 0)
			logMember = member;
		// 로그 추가
		accessRepo.save(AccessLog.builder()
						.member(logMember)
						.userAgent(userAgent)
						.remoteInfo(remoteInfo)
						.method(method)
						.requestURI(requestURI)
						.errorMsg(errorMsg)
						.build());
	}
	
	public void addLoginLog(Member member, boolean success, String userId, String remoteInfo, String socialAuth, String errorMsg) {
		// 로그인 시간 갱신
		LocalDateTime now = LocalDateTime.now();
		if(member != null) {
			member.setLastLoginTime(now);
			memberRepo.save(member);
		}
		
		// 로그 추가
		loginRepo.save(LoginLog.builder()
						.member(member)
						.success(success)
						.userId(userId)
						.remoteInfo(remoteInfo)
						.errorMsg(errorMsg)
						.socialAuth(socialAuth)
						.logTime(now)
						.build());
	}
	
	public void addFlowLog(Member member, String type, int count, String errorMsg) {
		flowRepo.save(FlowLog.builder()
						.member(member)
						.type(type)
						.count(count)
						.errorMsg(errorMsg)
						.build());
	}
	
	public void addTmsLog(Member member, String type, int count, String errorMsg) {
		tmsRepo.save(TmsLog.builder()
						.member(member)
						.type(type)
						.count(count)
						.errorMsg(errorMsg)
						.build());
	}
	
	public void addMemoLog(Member member, String type, int page, int count, long memoNo, String currentContent, String preContent) {
		memoRepo.save(MemoLog.builder()
						.member(member)
						.type(type)
						.page(page)
						.count(count)
						.memoNo(memoNo)
						.currentContent(currentContent)
						.preContent(preContent)
						.build());
	}
	
	public void addWeatherAPILog(String type, int originSize, int returnSize, int modifySize, String requestURI, String errorMsg) {
		weatherRepo.save(WeatherApiLog.builder()
						.logType(type)
						.originSize(originSize)
						.returnSize(returnSize)
						.modifySize(modifySize)
						.requestURI(requestURI)
						.errorMsg(errorMsg)
						.build());
	}
	
	public void addOutLierLog(String type, String prediectString) {
		outLierRepo.save(OutLierLog.builder()
						.type(type)
						.predictString(prediectString)
						.build());
	}
	
	public void addMailLog(Member member, String type, String errorMsg) {
		mailRepo.save(MailLog.builder()
						.member(member)
						.type(type)
						.errorMsg(errorMsg)
						.build());
	}
	
	public void addErrorLog(String source, String function, String errorMsg) {
		errorRepo.save(ErrorLog.builder()
						.errorSource(source)
						.errorFunction(function)
						.errorMsg(errorMsg)
						.build());
	}
}
