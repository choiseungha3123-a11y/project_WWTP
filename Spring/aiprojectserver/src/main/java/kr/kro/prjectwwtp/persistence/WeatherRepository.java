package kr.kro.prjectwwtp.persistence;
import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Weather;



public interface WeatherRepository extends JpaRepository<Weather, Long> {
	Weather findFirstByStnOrderByDataNoDesc(int stn);
	List<Weather> findByTimeAndStn(LocalDateTime time, int stn);
	List<Weather> findByTimeBetween(LocalDateTime start, LocalDateTime end);
	List<Weather> findByStnAndTimeBetween(int stn, LocalDateTime start, LocalDateTime end);
	//@Query(value = "SELECT * FROM tms_data WHERE stn = :stn AND time LIKE CONCAT(:time, '%') ORDER BY data_no ASC", nativeQuery = true)
	//List<TmsData> findByStnAndTime(int stn, String time);
	//Optional<TmsData> findByStnAndTime(int stn, LocalDateTime time);
}
