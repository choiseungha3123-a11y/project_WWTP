package kr.kro.prjectwwtp.util;

import java.util.Date;
import java.util.Optional;
import java.util.TimeZone;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.Claim;

import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import kr.kro.prjectwwtp.config.TokenBlacklistManager;
import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Role;
import kr.kro.prjectwwtp.persistence.MemberRepository;

public class JWTUtil {
	
	private static TokenBlacklistManager tokenBlacklistManager;
	private static MemberRepository memberRepo;
	//private static final long ACCESS_TOKEN_MSEC = 24 * 60 * (60 * 1000);	// 1일
	private static final long ACCESS_TOKEN_MSEC = 60 * (60 * 1000);	// 1시간
	//private static final long ACCESS_TOKEN_MSEC = (60 * 1000);	// 1분
	private static String JWT_KEY;
	
	public static final String prefix = "Bearer ";
	public static final String usernoClaim = "Userno";
	public static final String useridClaim = "Userid";
	public static final String usernameClaim = "Username";
	public static final String roleClaim = "Role";
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}
	
	public static void setKey(String key) {
		JWT_KEY = key;
	}
	
	public static void setTokenBlacklistManager(TokenBlacklistManager manager) {
		tokenBlacklistManager = manager;
	}
	
	public static void setMemberRepository(MemberRepository repo) {
		memberRepo = repo;
	}
	
	private static String getJWTSource(String token) {
		if(token.startsWith(prefix)) return token.replace(prefix, "");
		return token;
	}
	
	public static String getJWT(Long userno, String userid, String username, Role role) {
		//System.out.println("getJWT userno : " + userno);
		String src = JWT.create()
				.withClaim(usernoClaim, userno.toString())
				.withClaim(useridClaim, userid)
				.withClaim(usernameClaim, username)
				.withClaim(roleClaim, role.toString())
				.withExpiresAt(new Date(System.currentTimeMillis()+ACCESS_TOKEN_MSEC))
				.sign(Algorithm.HMAC256(JWT_KEY));
		return prefix + src;
	}
	public static String getJWT(Member member)
	{
		return getJWT(member.getUserNo(), member.getUserId(), member.getUserName(), member.getRole());
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
			
			// 블랙리스트에 있는 토큰인지 확인 (중복 로그인으로 무효화된 토큰)
			if (tokenBlacklistManager != null && tokenBlacklistManager.isTokenBlacklisted(tok)) {
				System.out.println("토큰이 블랙리스트에 있습니다 - 다른 기기에서의 로그인으로 무효화됨");
				return false;
			}
			
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
		boolean result = true;
		try {
			String token = request.getHeader("Authorization");
			result = isExpired(token);
		}
		catch(Exception e)
		{
			result = false;
		}
		return result;
	}
	
	public static Member parseToken(HttpServletRequest request) {
		String token = "none";
		try {
			token = request.getHeader("Authorization");
			if(isExpired(token))
				return null;
			Long userno = Long.parseLong(JWTUtil.getClaim(token, JWTUtil.usernoClaim));
			String userid = JWTUtil.getClaim(token, JWTUtil.useridClaim);
			Optional<Member> opt = memberRepo.findById(userno);
			if(opt.isEmpty())
				return null;
			Member member = opt.get();
			if(!member.getUserId().equals(userid))
				return null;
			return member;
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.out.println("token : " + token);
			System.out.println("쿠키 오류");
			return null;
		}
	}

}
