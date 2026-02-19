package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.ErrorLog;

public interface ErrorLogRepository extends JpaRepository<ErrorLog, Long>{

}