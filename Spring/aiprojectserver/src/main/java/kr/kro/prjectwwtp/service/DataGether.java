package kr.kro.prjectwwtp.service;

import java.net.URL;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.util.UriComponentsBuilder;

import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class DataGether {
	@Value("${spring.apihub.authKey}")
	private String apiKey;
	@Value("${spring.apihub.baseUrl}")
	private String baseUrl;

    @Bean
    String initGether() {
    	Map<String, Integer> stnValue = Stream.of(new Object[][] {
    		{"수택동", 368},
    		{"토평동", 569},
    		{"배양리", 541}
		}).collect(Collectors.toMap(item -> (String)item[0], item -> (Integer)item[1]));
    	String tm1 = "202401010000";
    	String tm2 = "202401020000";
//    	URL url = UriComponentsBuilder.fromUriString(baseUrl)
//    			.queryParam("tm1", tm1)
//    			.queryParam("tm2", tm2)
//    			.stn()
		System.out.println("apiKey : " + apiKey);
		return "initGether";
	}
}
