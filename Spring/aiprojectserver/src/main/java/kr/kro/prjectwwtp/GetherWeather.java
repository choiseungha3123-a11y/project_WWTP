package kr.kro.prjectwwtp;

import java.net.URI;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import jakarta.annotation.PostConstruct;
import kr.kro.prjectwwtp.domain.Weather;
import kr.kro.prjectwwtp.service.WeatherAPILogService;
import kr.kro.prjectwwtp.service.WeatherService;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class GetherWeather implements ApplicationRunner {
	private boolean isFirst = true;

	@Value("${spring.apihub.authKey}")
	private String authKey;
	@Value("${spring.apihub.baseUrl}")
	private String baseUrl;
	
	private final WeatherService weatherService;
	private final WeatherAPILogService logService;
	private RestTemplate restTemplate = new RestTemplate();
	
	@Value("${scheduler.enable}")
	private boolean enable;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}

	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
//		int fetchCount = fetchWeatherData();
//		while(fetchCount > 0)
//			fetchCount = fetchWeatherData();	
//		System.out.println("delayTerm : " + delayTerm);
//		System.out.println("enable : " + enable);
	}
	
	@Scheduled(fixedDelayString  = "${scheduler.gether.delay}") 
	public void fetchWeatherData() {
		if(isFirst) {
			isFirst = false;
			return;
		}
		if(!enable) return;
		
		int[] stnlist = { 368,		// 구리 수택동
				569, // 구리 토평동
				541 // 남양주 배양리
		};
		try {
			for (int stn : stnlist) {
				Weather lastData = weatherService.findFirstByStnOrderByLogTimeDesc(stn);
				LocalDateTime startTime = LocalDateTime.of(2024, 1, 1, 0, 0);
				if (lastData != null) {
					startTime = lastData.getLogTime();
				}
				LocalDateTime endTime = startTime.plusDays(1);
				String tm1 = startTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
				String tm2 = endTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
				List<Weather> list = fetchWeatherData(tm1, tm2, stn);
				// ta 값이 -99.9라면 예상값인걸로 보임
				list.removeIf(data -> data == null || data.getTa() == -99.9);
				if (list != null && list.size() > 0)
					weatherService.saveWeatherList(list);
				System.out.println("WeatherData 추가 : " + list.size());
			}	
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	public List<Weather> fetchWeatherData(String tm1, String tm2, int stn) {
		// build()와 expand()를 사용하여 값을 채워 넣습니다.
	    URI uri = UriComponentsBuilder.fromUriString(baseUrl)
	            .queryParam("tm1", tm1)
	            .queryParam("tm2", tm2)
	            .queryParam("stn", stn)
	            .queryParam("authKey", authKey)
	            .queryParam("disp", "0")
	            .build()            // 빌드
	            .toUri();           // URI 객체로 변환 (인코딩 포함)

	    // 실제 완성된 URL 확인
	    System.out.println("requrl : " + uri.toString()); 

	    // 호출 시 String이 아닌 URI 객체를 그대로 전달
	    String response = restTemplate.getForObject(uri, String.class);
	    
	    return parseResponse(response, uri.toString());
    }
	
	private List<Weather> parseResponse(String response, String uri) {
		//System.out.println("response : " + response);
        List<Weather> dataList = new ArrayList<>();
        if (response == null || response.isEmpty()) return dataList;

        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");
        
        int originSize = 0;
        int returnSize = 0;
        String errorMag = null;

        try {        	
	        // 3. 데이터 파싱 (주석 '#'으로 시작하는 줄 제외 및 공백 분리)
	        String[] lines = response.split("\n");
	        for (String line : lines) {
	            line = line.trim();
	            if (line.startsWith("#") || line.isEmpty()) continue;
	
	            String[] columns = line.split("\\s+"); // 공백 또는 탭으로 분리
	
	            try {
	            	++originSize;
	                // API 제공 순서에 맞춰 인덱스 매핑 (기상청 nph-aws2_min 사양 기준 예시)
	            	LocalDateTime tm = LocalDateTime.parse(columns[0], formatter);
	            	int stn = Integer.parseInt(columns[1]);
	            	double wd1 = Double.parseDouble(columns[2]);
	            	double wd2 = Double.parseDouble(columns[3]);
	            	double wds = Double.parseDouble(columns[4]);
	            	double wss = Double.parseDouble(columns[5]);
	            	double wd10 = Double.parseDouble(columns[6]);
	            	double ws10 = Double.parseDouble(columns[7]);
	            	double ts = Double.parseDouble(columns[8]);
	            	double re = Double.parseDouble(columns[9]);
	            	double rn15m = Double.parseDouble(columns[10]);
	            	double rn60m = Double.parseDouble(columns[11]);
	            	double rn12h = Double.parseDouble(columns[12]);
	            	double rnday = Double.parseDouble(columns[13]);
	            	double hm = Double.parseDouble(columns[14]);
	            	double pa = Double.parseDouble(columns[15]);
	            	double ps = Double.parseDouble(columns[16]);
	            	double td = Double.parseDouble(columns[17]);
	            	Weather data = Weather.builder()
	                        .logTime(tm) // TM
	                        .stn(stn)                // STN
	                        .wd1(wd1)              // WD1
	                        .wd2(wd2)              // WD2
	                        .wds(wds)              // WDS
	                        .wss(wss)              // WSS
	                        .wd10(wd10)             // WD10
	                        .ws10(ws10)             // WS10
	                        .ta(ts)               // TA
	                        .re(re)               // RE
	                        .rn15m(rn15m)           // RN_15M
	                        .rn60m(rn60m)           // RN_60M
	                        .rn12h(rn12h)           // RN_12H
	                        .rnday(rnday)           // RN_DAY
	                        .hm(hm)               // HM
	                        .pa(pa)               // PA
	                        .ps(ps)               // PS
	                        .td(td)               // TD
	                        .build();
	                
	                if(tm.isAfter(now))
	                	continue;
	                //if(ts)
	                //	continue;
	                dataList.add(data);
	                ++returnSize;
	            } catch (Exception e) {
	                // 데이터 결측치(-99.0 등)나 파싱 에러 처리
	                System.err.println("Line parsing error: " + line + " -> " + e.getMessage());
	            }
	        }
        } catch(Exception e) {
        	errorMag = e.getMessage();
        } finally {
			logService.addWeatherAPILog("Fetch", originSize,  returnSize, 0, uri, errorMag);
		}
        return dataList;
    }
}
