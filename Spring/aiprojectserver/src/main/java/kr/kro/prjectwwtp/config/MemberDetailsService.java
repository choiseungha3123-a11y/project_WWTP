package kr.kro.prjectwwtp.config;

import java.util.Optional;

import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemberDetailsService implements UserDetailsService {
	
	private final MemberRepository memberRepository;
	
	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		Optional<Member> optionalMember = memberRepository.findByUserId(username);
		
		if (!optionalMember.isPresent()) {
			throw new UsernameNotFoundException("사용자를 찾을 수 없습니다: " + username);
		}
		
		Member member = optionalMember.get();
		return new SecurityUser(member);
	}
}
