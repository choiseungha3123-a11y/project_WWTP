package kr.kro.prjectwwtp.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;


@Configuration
public class PasswordEncoder extends BCryptPasswordEncoder{
	public PasswordEncoder() {
		super();
	}
}