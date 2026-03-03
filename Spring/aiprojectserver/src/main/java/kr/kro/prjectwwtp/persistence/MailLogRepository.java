package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.MailLog;
import kr.kro.prjectwwtp.domain.Member;

public interface MailLogRepository extends JpaRepository<MailLog, Long>{
	void deleteByMember(Member member);
}