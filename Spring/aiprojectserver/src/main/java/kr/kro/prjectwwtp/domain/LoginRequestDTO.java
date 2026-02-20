package kr.kro.prjectwwtp.domain;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class LoginRequestDTO {
	private String userid;
	private String password;
	Role role;
}