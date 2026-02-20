package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.TmsPredict;

public interface TmsPredictRepository extends JpaRepository<TmsPredict, Long>{
	List<TmsPredict> findByTmsTimeBetweenOrderByTmsTimeAscTmsNoDesc(LocalDateTime start, LocalDateTime end);
}
