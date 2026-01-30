package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.AccessLog;

public interface AccessLogRepository extends JpaRepository<AccessLog, Long>{

}