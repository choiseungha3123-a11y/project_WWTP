package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.MemoLog;
import kr.kro.prjectwwtp.domain.TmsLog;

public interface TmsLogRepository extends JpaRepository<TmsLog, Long>{

}