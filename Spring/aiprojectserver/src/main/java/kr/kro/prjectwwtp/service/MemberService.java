package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import kr.kro.prjectwwtp.config.CryptoStringConverter;
import kr.kro.prjectwwtp.config.PasswordEncoder;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.AccessLogRepository;
import kr.kro.prjectwwtp.persistence.LoginLogRepository;
import kr.kro.prjectwwtp.persistence.MailLogRepository;
import kr.kro.prjectwwtp.persistence.MemberRepository;
import kr.kro.prjectwwtp.persistence.MemoLogRepository;
import kr.kro.prjectwwtp.persistence.MemoRepository;
import kr.kro.prjectwwtp.persistence.PublicRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemberService {
    private final CryptoStringConverter converter;
	private final MemberRepository memberRepo;
	private final AccessLogRepository accessLog;
	private final LoginLogRepository loginLog;
	private final MailLogRepository mailLog;
	private final MemoLogRepository memoLog;
	private final MemoRepository memoRepo;
	private final PublicRepository publicRepo;
	private PasswordEncoder encoder = new PasswordEncoder();
	
	public Member getByIdAndPassword(String userId, String password) {
		List<Member> list = memberRepo.findAll();
		Optional<Member> opt =  memberRepo.findByUserId(userId);
		if(opt.isEmpty()) {
			return null;
		}
		Member member = opt.get();
		if(!encoder.matches(password, member.getPassword())) {
			return null;
		}
		if(member.getDeleteTime() != null) {
			return null;
		}
		return member;
	}
	
	public Member findByNo(long userNo) {
		Optional<Member> opt = memberRepo.findById(userNo);
		if(opt.isEmpty()) {
			return null;
		}
		return opt.get();
	}
	
	public Member findById(String userId) {
		String encryptedUserId = converter.convertToDatabaseColumn(userId);
		Optional<Member> opt = memberRepo.findByUserId(encryptedUserId);
		if(opt.isEmpty()) {
			return null;
		}
		return opt.get();
	}
	
	public boolean checkId(String userId) {
		return memberRepo.findByUserId(userId).isPresent();
	}
	
	public boolean checkEmail(String userEmail) {
		return memberRepo.findByUserEmail(userEmail).isPresent();
	}
	
	public List<Member> getMemberList() {
		return memberRepo.findAllByDeleteTimeIsNull();
		//return memberRepo.findAll();
	}
	
	public void saveAll(List<Member> list) {
		memberRepo.saveAll(list);
	}
	
	public void addMember(String userId, String password, String userName, String userEmail) {
		memberRepo.save(Member.builder()
				.userId(userId)
				.password(encoder.encode(password))
				.userName(userName)
				.userEmail(userEmail)
				.role(Role.ROLE_MEMBER)
				.build());
	}
	
	public void modifyMember(Member member, String userId, String password, String userName, String userEmail, Role role) {
		if(userId != null && userId.length() > 0)
			member.setUserId(userId);
		if(password != null && password.length() > 0)
			member.setPassword(encoder.encode(password));
		if(userName != null && userName.length() > 0)
			member.setUserName(userName);
		if(userEmail != null && userEmail.length() > 0) {
			member.setUserEmail(userEmail);
			member.setValidateEmail(false);
			member.setValidateKey(null);
		}
		if(role != null)
			member.setRole(role);
		memberRepo.save(member);
	}
	
	public Member addSocialMember(String socialAuth, String userId, String userName) {
		Member member = Member.builder()
				.userName(userName)
				.userId(userId)
				.password("socialUser")
				.socialAuth(socialAuth)
				.role(Role.ROLE_VIEWER)
				.build();
		memberRepo.save(member);
		return member;
	}
	
	public Member findBySocialAuth(String socialAuth) {
		Optional<Member> opt = memberRepo.findBySocialAuth(socialAuth); 
		if(opt.isEmpty())
			return null;
		return opt.get();
	}
	
	public void deleteMember(Member member) {
		// 블라인드 처리
		member.setDeleteTime(LocalDateTime.now());
		memberRepo.save(member);
	}
	
	@Transactional
	public void deleteFromDB(Member member) {
		// DB에서 완전 삭제
		accessLog.deleteByMember(member);
		loginLog.deleteByMember(member);
		mailLog.deleteByMember(member);
		memoLog.deleteByMember(member);	
		memoRepo.deleteByCreateMember(member);
		memoRepo.deleteByModifyMember(member);
		memoRepo.deleteByDisableMember(member);
		publicRepo.deleteByMember(member);
		
		memberRepo.deleteById(member.getUserNo());
	}
	
	public void addEmailKey(Long userNo, String key) {
		Member member = findByNo(userNo);
		member.setValidateKey(key);
		memberRepo.save(member);
	}
	
	public void validEmail(Long userNo) {
		Member member = findByNo(userNo);
		member.setValidateEmail(true);
		member.setValidateKey(null);
		memberRepo.save(member);
	}
	
	public void delteEmail(Long userNo) {
		Member member = findByNo(userNo);
		member.setValidateEmail(false);
		member.setValidateKey(null);
		member.setUserEmail(null);
		memberRepo.save(member);
	}
	
	public List<Member> getValidateEmailMember() {
		return memberRepo.findByUserEmailIsNotNullAndValidateKeyIsNullAndValidateEmailTrue();
	}

}
