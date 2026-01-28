package kr.kro.prjectwwtp.persistence;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Member;
import java.util.List;
import kr.kro.prjectwwtp.domain.Role;




public interface MemberRepository extends JpaRepository<Member, Long> {
	Optional<Member> findByUserId(String userId);
	List<Member> findByRole(Role role);
	
}
