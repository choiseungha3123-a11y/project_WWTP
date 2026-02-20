package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.TmsOrigin;

public interface TmsOriginRepository extends JpaRepository<TmsOrigin, Long>{
	List<TmsOrigin> findByTmsTimeBetween(LocalDateTime start, LocalDateTime end);
}
