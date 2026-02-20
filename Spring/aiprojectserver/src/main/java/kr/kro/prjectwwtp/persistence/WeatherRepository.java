package kr.kro.prjectwwtp.persistence;
import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Weather;



public interface WeatherRepository extends JpaRepository<Weather, Long> {
	Weather findFirstByStnOrderByLogTimeDesc(int stn);
	List<Weather> findByLogTimeAndStn(LocalDateTime time, int stn);
	List<Weather> findByLogTimeBetween(LocalDateTime start, LocalDateTime end);
	List<Weather> findByStnAndLogTimeBetween(int stn, LocalDateTime start, LocalDateTime end);
}
