package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.WeatherComplete;


public interface WeatherCompleteRepository extends JpaRepository<WeatherComplete, Long> {
	WeatherComplete findFirstByStnOrderByDataNoDesc(int stn);
}
