package kr.kro.prjectwwtp.persistence;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.TmsData;


public interface DataRepository extends JpaRepository<TmsData, Long> {
}
