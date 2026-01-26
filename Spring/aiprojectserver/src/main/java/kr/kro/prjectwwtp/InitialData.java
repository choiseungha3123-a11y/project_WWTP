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
		if(memberRepo.findByUserid(adminUserid).isEmpty()) {
			memberRepo.save(Member.builder()
					.userid(adminUserid)
					.password(encoder.encode("admin1234"))
					.role(Role.ROLE_ADMIN)
					.build());
		}
	}
}
	
	
	
