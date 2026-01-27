package kr.kro.prjectwwtp;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class InitialData implements ApplicationRunner {
	private final MemberRepository memberRepo;
	private PasswordEncoder encoder = new PasswordEncoder();
	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
		//System.out.println("InitalData");
		String adminUserid = "admin";
		String memberUserid = "member";
		if(memberRepo.findByUserId(adminUserid).isEmpty()) {
			memberRepo.save(Member.builder()
					.userId(adminUserid)
					.password(encoder.encode("admin1234"))
					.userName("관리자")
					.role(Role.ROLE_ADMIN)
					.build());
		}
		if(memberRepo.findByUserId(memberUserid).isEmpty()) {
			memberRepo.save(Member.builder()
					.userId(memberUserid)
					.password(encoder.encode("member1234"))
					.userName("이용자")
					.role(Role.ROLE_ADMIN)
					.build());
		}
	}
}
	
	
	
