package kr.kro.prjectwwtp.config;

import org.springframework.security.core.SpringSecurityCoreVersion;
import org.springframework.security.core.authority.AuthorityUtils;
import org.springframework.security.core.userdetails.User;

import kr.kro.prjectwwtp.domain.Member;

public class SecurityUser extends User{
	private static final long serialVersionUID = SpringSecurityCoreVersion.SERIAL_VERSION_UID;
	private Member member;
	
	public SecurityUser(Member member) {
		super(member.getUserid(), member.getPassword(), AuthorityUtils.createAuthorityList(member.getRole().toString()));
		this.member = member;
	}
	
	public Member getMember() {
		return member;
	}

}
