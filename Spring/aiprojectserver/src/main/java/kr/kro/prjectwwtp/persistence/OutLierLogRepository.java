package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.OutLierLog;

public interface OutLierLogRepository extends JpaRepository<OutLierLog, Long>{

}