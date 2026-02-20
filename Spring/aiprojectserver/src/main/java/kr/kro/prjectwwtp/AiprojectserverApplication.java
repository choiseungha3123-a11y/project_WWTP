package kr.kro.prjectwwtp;

import java.time.LocalDateTime;
import java.util.TimeZone;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class AiprojectserverApplication {

	public static void main(String[] args) {
		TimeZone.setDefault(TimeZone.getTimeZone("${spring.timezone}"));
		SpringApplication.run(AiprojectserverApplication.class, args);
		System.out.println(LocalDateTime.now());
	}

}
