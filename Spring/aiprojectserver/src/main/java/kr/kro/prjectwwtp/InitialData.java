package kr.kro.prjectwwtp;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.controller.TmsController;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.util.JWTUtil;
import kr.kro.prjectwwtp.util.Util;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class InitialData implements ApplicationRunner {
	private final MemberRepository memberRepo;
	private final TmsController tmsController;
	private PasswordEncoder encoder = new PasswordEncoder();
	
	@Value("${jwt.key}")
	private String setJWTKey;
	@Value("${util.key}")
	private String setUtilKey;
	
	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
		//System.out.println("InitalData");
		
		JWTUtil.setKey(setJWTKey);
		Util.setKey(setUtilKey);
		
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
		tmsController.makeFakeDate();
	}
}
	
	
	
