package kr.kro.prjectwwtp.persistence;
import java.time.LocalDateTime;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.WeatherComplete;



public interface WeatherCompleteRepository extends JpaRepository<WeatherComplete, Long> {
	WeatherComplete findFirstByStnOrderByDataNoDesc(int stn);
	boolean existsByStnAndDataTime(int stn, LocalDateTime dataTime);
}
