package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FlowLog;

public interface FlowLogRepository extends JpaRepository<FlowLog, Long>{

}