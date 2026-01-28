package kr.kro.prjectwwtp.service;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemberService {
	private final MemberRepository memberRepo;
	private PasswordEncoder encoder = new PasswordEncoder();
	
	public Member getByIdAndPassword(String userId, String password) {
		Optional<Member> opt =  memberRepo.findByUserId(userId);
		if(opt.isEmpty()) {
			return null;
		}
		Member member = opt.get();
		if(!encoder.matches(password, member.getPassword())) {
			return null;
		}
		return member;
	}
	
	public Member getByNo(long userNo) {
		Optional<Member> opt = memberRepo.findById(userNo);
		if(opt.isEmpty()) {
			return null;
		}
		return opt.get();
	}
	
	public boolean checkId(String userId) {
		return memberRepo.findByUserId(userId).isPresent();
	}
	
	public List<Member> getMemberList() {
		return memberRepo.findAll();
	}
	
	public void addMember(String userId, String password, String userName) {
		memberRepo.save(Member.builder()
				.userId(userId)
				.password(encoder.encode(password))
				.userName(userName)
				.role(Role.ROLE_MEMBER)
				.build());
	}
	
	public void modifyMember(Member member, String userId, String password, String userName, Role role) {
		if(userId != null && userId.length() > 0)
			member.setUserId(userId);
		if(password != null && password.length() > 0)
			member.setPassword(encoder.encode(password));
		if(userName != null && userName.length() > 0)
			member.setUserName(userName);
		if(role != null)
			member.setRole(role);
		memberRepo.save(member);
	}
	
	public void deleteMember(Member member) {
		memberRepo.delete(member);
	}

}
