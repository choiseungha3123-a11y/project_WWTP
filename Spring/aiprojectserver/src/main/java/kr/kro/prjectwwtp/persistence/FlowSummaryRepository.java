package kr.kro.prjectwwtp.persistence;
import java.util.Date;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FlowSummary;



public interface FlowSummaryRepository extends JpaRepository<FlowSummary, Date> {
	
}
