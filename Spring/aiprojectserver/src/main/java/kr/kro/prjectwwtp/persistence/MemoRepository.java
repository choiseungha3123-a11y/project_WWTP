package kr.kro.prjectwwtp.persistence;
import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Memo;



public interface MemoRepository extends JpaRepository<Memo, Long> {
	Optional<Memo> findByMemoNoAndDisableMemberIsNull(long memoNo);
	Page<Memo> findByDisableMemberIsNotNull(Pageable pageable);
	Page<Memo> findByDisableMemberIsNull(Pageable pageable);
}
