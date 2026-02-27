package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.WeatherSummary;
import kr.kro.prjectwwtp.domain.WeatherSummaryId;



public interface WeatherSummaryRepository extends JpaRepository<WeatherSummary, WeatherSummaryId> {
	
}
