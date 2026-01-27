package kr.kro.prjectwwtp.persistence;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.WeatherComplete;


public interface WeatherCompleteRepository extends JpaRepository<WeatherComplete, Long> {
	List<WeatherComplete> findByStnOrderByDataNoDesc(int stn);
}
