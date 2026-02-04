package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.TmsImputate;

public interface TmsImputateRepository extends JpaRepository<TmsImputate, Long>{
	List<TmsImputate> findByTmsTimeBetween(LocalDateTime start, LocalDateTime end);
	boolean existsByTmsTime(LocalDateTime time);
}
