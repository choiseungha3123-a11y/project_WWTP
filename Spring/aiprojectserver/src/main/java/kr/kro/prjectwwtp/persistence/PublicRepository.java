package kr.kro.prjectwwtp.persistence;


import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.PublicDTO;

public interface PublicRepository extends JpaRepository<PublicDTO, Long>{
	Page<PublicDTO> findByDeleteTimeIsNull(Pageable pageable);
	void deleteByMember(Member member);
}