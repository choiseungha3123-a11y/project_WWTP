package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.WeatherApiLog;



public interface WeatherAPILogRepository extends JpaRepository<WeatherApiLog, Long> {
}
