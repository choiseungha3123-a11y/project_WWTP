package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import kr.kro.prjectwwtp.domain.TmsData;
import java.util.List;
import java.util.Optional;
import java.time.LocalDateTime;



public interface WeatherRepository extends JpaRepository<TmsData, Long> {
	TmsData findFirstByStnOrderByDataNoDesc(int stn);
	//TmsData findFirstByOrderByDataNoDesc(); 
	List<TmsData> findByTimeAndStn(LocalDateTime time, int stn);
	List<TmsData> findByTimeBetweenOrderByDataNoDesc(LocalDateTime start, LocalDateTime end);
	List<TmsData> findByStnAndTimeBetweenOrderByDataNoDesc(int stn, LocalDateTime start, LocalDateTime end);
	@Query(value = "SELECT * FROM tms_data WHERE stn = :stn AND time LIKE CONCAT(:time, '%') ORDER BY data_no ASC", nativeQuery = true)
	List<TmsData> findByStnAndTime(int stn, String time);
	Optional<TmsData> findByStnAndTime(int stn, LocalDateTime time);
}
