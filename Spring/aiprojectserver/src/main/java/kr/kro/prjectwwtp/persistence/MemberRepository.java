package kr.kro.prjectwwtp.persistence;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.Member;


public interface MemberRepository extends JpaRepository<Member, Long> {
	Optional<Member> findByUserid(String userid);
}
