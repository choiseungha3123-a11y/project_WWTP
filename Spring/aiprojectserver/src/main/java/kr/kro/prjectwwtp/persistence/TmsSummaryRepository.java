package kr.kro.prjectwwtp.persistence;
import java.time.LocalDateTime;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.TmsSummary;



public interface TmsSummaryRepository extends JpaRepository<TmsSummary, LocalDateTime> {
	
}
