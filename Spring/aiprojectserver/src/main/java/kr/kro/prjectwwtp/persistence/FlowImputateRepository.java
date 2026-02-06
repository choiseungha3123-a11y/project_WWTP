package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FlowImputate;

public interface FlowImputateRepository extends JpaRepository<FlowImputate, Long>{
	List<FlowImputate> findByFlowTimeBetweenOrderByFlowTime(LocalDateTime start, LocalDateTime end);
	boolean existsByFlowTime(LocalDateTime time);
}
