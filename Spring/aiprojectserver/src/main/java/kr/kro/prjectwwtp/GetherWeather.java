package kr.kro.prjectwwtp;

import java.net.URI;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class GetherWeather implements ApplicationRunner {

	@Value("${spring.apihub.authKey}")
	private String authKey;
	@Value("${spring.apihub.baseUrl}")
	private String baseUrl;
	
	private final WeatherRepository weatherRepo;
	private RestTemplate restTemplate = new RestTemplate();
	
	@Value("${scheduler.delay_term}")
	private int delayTerm;
	private int delayCount = 0;
	private int fetchListCount = -1;
	@Value("${scheduler.delay}")
	private int delaytime; 
	@Value("${scheduler.enable}")
	private boolean enable;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
//		int fetchCount = fetchTmsData();
//		while(fetchCount > 0)
//			fetchCount = fetchTmsData();	
//		System.out.println("delayTerm : " + delayTerm);
//		System.out.println("enable : " + enable);
	}
	
	@Scheduled(fixedDelayString  = "${scheduler.delay}") 
	public void fetchTmsData() {
		if(!enable) return;
		if(fetchListCount == 0)
		{
			++delayCount;
			if(delayCount == delayTerm / delaytime)
			{
				fetchListCount = -1;
				System.out.println("30 minute delayed");
			}
			return;
		}
		System.out.println("fetch start");
		delayCount = 0;
		fetchListCount = 0;
		int[] stnlist = { 368,		// 구리 수택동
				569, // 구리 토평동
				541 // 남양주 배양리
		};
		try {
			for (int stn : stnlist) {
				TmsData lastData = weatherRepo.findFirstByStnOrderByDataNoDesc(stn);
				LocalDateTime startTime = LocalDateTime.of(2024, 1, 1, 0, 0);
				if (lastData != null) {
					startTime = lastData.getTime();
				}
				LocalDateTime endTime = startTime.plusDays(1);
				String tm1 = startTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
				String tm2 = endTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
				List<TmsData> list = fetchTmsData(tm1, tm2, stn);
				// ta 값이 -99.9라면 예상값인걸로 보임
				list.removeIf(data -> data == null || data.getTa() == -99.9);
				if (list != null && list.size() > 0)
					weatherRepo.saveAll(list);
				fetchListCount += list.size();
			}	
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
			fetchListCount = -1;
		}
	}

	public List<TmsData> fetchTmsData(String tm1, String tm2, int stn) {
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
	    
	    return parseResponse(response);
    }
	
	private List<TmsData> parseResponse(String response) {
		//System.out.println("response : " + response);
        List<TmsData> dataList = new ArrayList<>();
        if (response == null || response.isEmpty()) return dataList;

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");

        // 3. 데이터 파싱 (주석 '#'으로 시작하는 줄 제외 및 공백 분리)
        String[] lines = response.split("\n");
        for (String line : lines) {
            line = line.trim();
            if (line.startsWith("#") || line.isEmpty()) continue;

            String[] columns = line.split("\\s+"); // 공백 또는 탭으로 분리

            try {
                // API 제공 순서에 맞춰 인덱스 매핑 (기상청 nph-aws2_min 사양 기준 예시)
                TmsData data = TmsData.builder()
                        .time(LocalDateTime.parse(columns[0], formatter)) // TM
                        .stn(Integer.parseInt(columns[1]))                // STN
                        .wd1(Double.parseDouble(columns[2]))              // WD1
                        .wd2(Double.parseDouble(columns[3]))              // WD2
                        .wds(Double.parseDouble(columns[4]))              // WDS
                        .wss(Double.parseDouble(columns[5]))              // WSS
                        .wd10(Double.parseDouble(columns[6]))             // WD10
                        .ws10(Double.parseDouble(columns[7]))             // WS10
                        .ta(Double.parseDouble(columns[8]))               // TA
                        .re(Double.parseDouble(columns[9]))               // RE
                        .rn15m(Double.parseDouble(columns[10]))           // RN_15M
                        .rn60m(Double.parseDouble(columns[11]))           // RN_60M
                        .rn12h(Double.parseDouble(columns[12]))           // RN_12H
                        .rnday(Double.parseDouble(columns[13]))           // RN_DAY
                        .hm(Double.parseDouble(columns[14]))               // HM
                        .pa(Double.parseDouble(columns[15]))               // PA
                        .ps(Double.parseDouble(columns[16]))               // PS
                        .td(Double.parseDouble(columns[17]))               // TD
                        .build();
                
                dataList.add(data);
            } catch (Exception e) {
                // 데이터 결측치(-99.0 등)나 파싱 에러 처리
                System.err.println("Line parsing error: " + line + " -> " + e.getMessage());
            }
        }
        return dataList;
    }
}
