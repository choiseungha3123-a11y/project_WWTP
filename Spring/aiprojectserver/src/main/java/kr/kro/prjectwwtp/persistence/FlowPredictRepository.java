package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FlowPredict;

public interface FlowPredictRepository extends JpaRepository<FlowPredict, Long>{
	List<FlowPredict> findByFlowTimeBetweenOrderByFlowTimeAscFlowNoDesc(LocalDateTime start, LocalDateTime end);
}
