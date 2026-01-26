package kr.kro.prjectwwtp.util;

import java.util.Date;
import java.util.Optional;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.Claim;

import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;

public class JWTUtil {
	//private static final long ACCESS_TOKEN_MSEC = 24 * 60 * (60 * 1000);	// 1일
	private static final long ACCESS_TOKEN_MSEC = 60 * (60 * 1000);	// 1시간
	//private static final long ACCESS_TOKEN_MSEC = (60 * 1000);	// 1분
	private static final String JWT_KEY = "kr.kro.projectwwtp";
	
	public static final String prefix = "Bearer ";
	public static final String usernoClaim = "Userno";
	public static final String useridClaim = "Userid";
	public static final String roleClaim = "Role";
	
	private static String getJWTSource(String token) {
		if(token.startsWith(prefix)) return token.replace(prefix, "");
		return token;
	}
	
	public static String getJWT(Long userno, String userid, Role role) {
		//System.out.println("getJWT userno : " + userno);
		String src = JWT.create()
				.withClaim(usernoClaim, userno.toString())
				.withClaim(useridClaim, userid)
				.withClaim(roleClaim, role.toString())
				.withExpiresAt(new Date(System.currentTimeMillis()+ACCESS_TOKEN_MSEC))
				.sign(Algorithm.HMAC256(JWT_KEY));
		return prefix + src;
	}
	public static String getJWT(Member member)
	{
		return getJWT(member.getUser_no(), member.getUserid(), member.getRole());
	}
	
	// JWT에서 Claim 추출할 때 호출
	public static String getClaim(String token, String cname) {
		String tok = getJWTSource(token);
		Claim claim = JWT.require(Algorithm.HMAC256(JWT_KEY)).build()
						.verify(tok).getClaim(cname);
		if (claim.isMissing() || claim.isNull()) return null;
		return claim.asString();
	}
	
	private static boolean isExpired(String token) {
		boolean result = true;
		try {
			String tok = getJWTSource(token);
			result = JWT.require(Algorithm.HMAC256(JWT_KEY)).build()
							.verify(tok).getExpiresAt().before(new Date());
		}catch(Exception e)
		{
			//e.printStackTrace();
			System.out.println("쿠키 만료");
			result = false;
		}
		return result;
	}
	
	public static boolean isExpired(HttpServletRequest request)
	{
		try {
			String token = request.getHeader("Authorization");
			if(isExpired(token))
				return true;
		}
		catch(Exception e)
		{
			return true;
		}
		return false;
	}
	
	public static Member parseToken(HttpServletRequest request, MemberRepository memberRepo) {
		try {
		String token = request.getHeader("Authorization");
		if(isExpired(token))
			return null;
		Long userno = Long.parseLong(JWTUtil.getClaim(token, JWTUtil.usernoClaim));
		String userid = JWTUtil.getClaim(token, JWTUtil.useridClaim);
		Optional<Member> opt = memberRepo.findById(userno);
		if(opt.isEmpty())
			return null;
		Member member = opt.get();
		if(!member.getUserid().equals(userid))
			return null;
		return member;
		}
		catch(Exception e)
		{
			//e.printStackTrace();
			System.out.println("쿠키 오류");
			return null;
		}
	}

}
