package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FlowOrigin;

public interface FlowOriginRepository extends JpaRepository<FlowOrigin, Long>{
	List<FlowOrigin> findByFlowTimeBetween(LocalDateTime start, LocalDateTime end);
}
