package kr.kro.prjectwwtp.persistence;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Memo;



public interface MemoRepository extends JpaRepository<Memo, Long> {
	
}
