package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.LoginLog;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.persistence.LoginLogRepository;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class LoginLogService {
	private final MemberRepository memberRepo;
	private final LoginLogRepository logRepo;
	
	public void addLog(Member member) {
		// 로그인 시간 갱신
		LocalDateTime now = LocalDateTime.now();
		member.setLastLoginTime(now);
		memberRepo.save(member);
		
		// 로그 추가
		logRepo.save(LoginLog.builder()
				.member(member)
				.loginTime(now)
				.build());
	}

}
