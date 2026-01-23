package kr.kro.prjectwwtp;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Component;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.persistence.DataRepository;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class DataGether implements ApplicationRunner {
	@Value("${spring.apihub.authKey}")
	private String apiKey;
	@Value("${spring.apihub.baseUrl}")
	private String baseUrl;
	
	private final DataRepository dataRepo;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		TmsData lastData = getLastData();
		// TODO Auto-generated method stub
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
	}
    @Bean
    String initGether() {
    	
		return "initGether";
	}
    
    TmsData getLastData() {
    	Pageable pageable = PageRequest.of(0,  10, Sort.Direction.DESC, "data_no");
    	Page<TmsData> page = dataRepo.findAll(pageable);
    	if(page.isEmpty())
    		return null;
    	return page.getContent().get(0);
    	
    }
}
