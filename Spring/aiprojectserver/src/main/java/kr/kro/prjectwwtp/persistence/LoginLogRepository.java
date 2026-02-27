package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.LoginLog;
import kr.kro.prjectwwtp.domain.Member;

public interface LoginLogRepository extends JpaRepository<LoginLog, Long>{
	void deleteByMember(Member member);

}