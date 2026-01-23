package kr.kro.prjectwwtp.service;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class InitialData {
	private final MemberRepository memberRepo;
	

	@Bean
	String initData() {
		String adminUserid = "admin";
		if(memberRepo.findByUserid(adminUserid).isEmpty()) {
			memberRepo.save(Member.builder()
					.userid(adminUserid)
					.password("admin1234")
					.role(Role.ROLE_ADMIN)
					.build());
		}
		return "initData";
	}
}
