package kr.kro.prjectwwtp.persistence;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;

import java.util.List;



public interface MemoRepository extends JpaRepository<Memo, Long> {
	
}
