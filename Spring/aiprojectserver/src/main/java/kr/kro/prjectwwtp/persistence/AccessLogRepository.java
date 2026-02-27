package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.AccessLog;
import kr.kro.prjectwwtp.domain.Member;


public interface AccessLogRepository extends JpaRepository<AccessLog, Long>{
	void deleteByMember(Member member);
}