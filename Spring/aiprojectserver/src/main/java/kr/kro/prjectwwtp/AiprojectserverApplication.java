package kr.kro.prjectwwtp;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class AiprojectserverApplication {

	public static void main(String[] args) {
		SpringApplication.run(AiprojectserverApplication.class, args);
	}

}
