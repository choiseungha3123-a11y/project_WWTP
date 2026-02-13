package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.MailLog;

public interface MailLogRepository extends JpaRepository<MailLog, Long>{

}