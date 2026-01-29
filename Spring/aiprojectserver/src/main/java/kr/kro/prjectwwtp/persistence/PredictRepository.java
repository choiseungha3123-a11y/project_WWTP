package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Predict;



public interface PredictRepository extends JpaRepository<Predict, Long> {
	
}
