package kr.kro.prjectwwtp;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.controller.TmsOriginController;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class InitialData implements ApplicationRunner {
	private final MemberRepository memberRepo;
	private final TmsOriginController tmsController;
	private PasswordEncoder encoder = new PasswordEncoder();
	
	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
		//System.out.println("InitalData");
		String adminUserid = "admin";
		String memberUserid = "member";
		if(memberRepo.findByRole(Role.ROLE_ADMIN).size() == 0) {
			memberRepo.save(Member.builder()
					.userId(adminUserid)
					.password(encoder.encode("admin1234"))
					.userName("관리자")
					.role(Role.ROLE_ADMIN)
					.build());
		}
		if(memberRepo.findByRole(Role.ROLE_MEMBER).size() == 0) {
			memberRepo.save(Member.builder()
					.userId(memberUserid)
					.password(encoder.encode("member1234"))
					.userName("이용자")
					.role(Role.ROLE_MEMBER)
					.build());
		}
		tmsController.postMakeFakeDate();
	}
	
	@Scheduled(cron = "${scheduler.fakeday.cron}")
	public void makeFakeDate() {
		tmsController.postMakeFakeDate();
	}
}
	
	
	
