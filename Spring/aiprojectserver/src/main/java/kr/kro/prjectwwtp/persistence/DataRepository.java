package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.TmsData;
import java.util.List;
import java.time.LocalDateTime;



public interface DataRepository extends JpaRepository<TmsData, Long> {
	TmsData findFirstByStnOrderByDataNoDesc(int stn);
	//TmsData findFirstByOrderByDataNoDesc(); 
	List<TmsData> findByTimeAndStn(LocalDateTime time, int stn);
}
