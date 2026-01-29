package kr.kro.prjectwwtp;

import java.net.URI;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.domain.WeatherComplete;
import kr.kro.prjectwwtp.persistence.WeatherCompleteRepository;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class CompleteWeather implements ApplicationRunner {

	@Value("${spring.apihub.authKey}")
	private String authKey;
	@Value("${spring.apihub.baseUrl}")
	private String baseUrl;
	
	private final WeatherRepository weatherRepo;
	private final WeatherCompleteRepository completeRepo;
	private final GetherWeather getherWeather;
	private RestTemplate restTemplate = new RestTemplate();
		@Value("${scheduler.enable}")
	private boolean enable;

	@Override
	public void run(ApplicationArguments args) throws Exception {
	}
	
	boolean compareDay(LocalDateTime a, LocalDateTime b) {
		if(a.getYear() == b.getYear() && a.getMonth() == b.getMonth() && a.getDayOfMonth() == a.getDayOfMonth())
			return true;
		return false;
	}
	
	List<TmsData> getList(int stn, LocalDateTime date) {
		LocalDateTime start = LocalDateTime.of(date.getYear(), date.getMonth(), date.getDayOfMonth(), 0, 0);
		LocalDateTime end = LocalDateTime.of(date.getYear(), date.getMonth(), date.getDayOfMonth(), 23, 59).with(LocalTime.MAX);
		List<TmsData> list = weatherRepo.findByStnAndTimeBetween(stn, start, end);
		
		return list;
	}
	
	@Scheduled(fixedDelayString  = "${scheduler.delay}") 
	public void completeWeatherData() {
		if(!enable) return;
		
		int[] stnlist = { 368,		// 구리 수택동
				569, // 구리 토평동
				541 // 남양주 배양리
		};
		try {
			for(int stn : stnlist) {
				WeatherComplete complete = completeRepo.findFirstByStnOrderByDataNoDesc(stn);
				LocalDateTime last = LocalDateTime.of(2024, 1, 1, 0, 0);
				int dataCount = 0;
				LocalDateTime now = LocalDateTime.now();
				//System.out.println("now : " + now);
				if(complete != null) {
					last = complete.getDataTime();
					dataCount = complete.getDataSize();
				}
				
				while(last.isBefore(now) || dataCount != 24 * 60) {
					LocalDateTime start = LocalDateTime.of(last.getYear(), last.getMonthValue(), last.getDayOfMonth(), 0, 0);
					LocalDateTime end = LocalDateTime.of(last.getYear(), last.getMonthValue(), last.getDayOfMonth(), 23, 59);
					List<TmsData> list = weatherRepo.findByStnAndTimeBetween(stn, start, end);
					
					int size = list.size();
					if(size == 24 * 60) {
						// 데이터 완료
						last = last.plusDays(1);
						completeRepo.save(WeatherComplete.builder()
											.dataTime(start)
											.stn(stn)
											.dataSize(size)
											.build());
					} else if(size > 24 * 60) {
						// 중복 발견 - time값이 동일한 중복 데이터 제거
						System.out.println("중복 발견 : " + last);
						List<TmsData> toDelete = new ArrayList<>();
						var uniqueMap = list.stream()
							.collect(Collectors.toMap(
								TmsData::getTime,
								d -> d,
								(existing, duplicate) -> {
									toDelete.add(duplicate);
									return existing;
								}
							));
						
						// 데이터베이스에서 삭제
						weatherRepo.deleteAll(toDelete);
						
						System.out.println("중복 제거 완료: " + toDelete.size() + "개 삭제");
						list = new ArrayList<>(uniqueMap.values());
					} else {
						// 결측 발견
						System.out.println("결측 발견 : " + last + " (현재: " + size + "개)");
						
						// GetherWeather를 이용해 API에서 데이터 조회
						String tm1 = start.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
						String tm2 = end.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
						
						List<TmsData> apiList = getherWeather.fetchTmsData(tm1, tm2, stn);
						System.out.println("API에서 조회: " + apiList.size() + "개");
						
						// 현재 list에 없는 데이터 찾기 (현재 list의 time 값과 비교)
						Set<LocalDateTime> existingTimes = list.stream()
							.map(TmsData::getTime)
							.collect(Collectors.toSet());
						
						List<TmsData> missingData = apiList.stream()
							.filter(data -> !existingTimes.contains(data.getTime()))
							.collect(Collectors.toList());
						
						// 빠진 데이터를 DB에 저장
						if(missingData.size() > 0) {
							weatherRepo.saveAll(missingData);
							System.out.println("결측 데이터 보충: " + missingData.size() + "개 저장");
							list.addAll(missingData);
						} else {
							System.out.println("추가될 결측 데이터 없음");
						}
					}
				}
				System.out.println("last : " + last);
			}
/*			
			for (int stn : stnlist) {
				LocalDateTime lastDay = LocalDateTime.of(2024, 1, 1, 0, 0);
				LocalDateTime now = LocalDateTime.now();
				//List<WeatherComplete> listAll = completeRepo.findByStnOrderByDataNoDesc(stn);
				List<WeatherComplete> listAll = completeRepo.findByStnOrderByDataNoDesc(stn);
				for(WeatherComplete c : listAll) {
					if(c.getDataSize() != 24 * 60 && !compareDay(now, c.getDataTime()))
					{
						List<TmsData> list = getList(c.getStn(), c.getDataTime());
						// 이상 및 결측 발견
						if(c.getDataSize() > 24 * 60) {
							// 중첩
							System.out.println("중첩 발견");
							while(list.size() > 24 * 60)
							{
								TmsData match = list.stream()
								    .filter(data -> data.getTime().equals(c.getDataTime()))
								    .findFirst()
								    .orElse(null);
								if(match != null) {
									list.remove(match);
									weatherRepo.delete(match);
								}
							}
							System.out.println("size : " + list.size());
							System.out.println("중첩 제거");
							c.setDataSize(list.size());
							completeRepo.save(c);
						} 
						else if(c.getDataSize() < 24 * 60) {
							// 결측
//							LocalDateTime checkTime = c.getDataTime();
//							LocalDateTime startTime = LocalDateTime.of(checkTime.getYear(), checkTime.getMonthValue(), checkTime.getDayOfMonth(), 0, 0);
//							LocalDateTime endTime = startTime.plusDays(1);
//							String tm1 = startTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
//							String tm2 = endTime.format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
//							List<TmsData> newlist = fetchTmsData(tm1, tm2, stn);
//							for(TmsData data : newlist) {
//								weatherRepo.findby
//							}
						}
					}
					lastDay = c.getDataTime();
				}
				while(lastDay.isBefore(now)) {
					// 오늘 이전의 처리가 안된 데이터가 있음
					List<TmsData> list = getList(stn, lastDay);
					System.out.println(lastDay + " >> list 개수 : " + list.size());
					completeRepo.save(WeatherComplete.builder()
											.dataTime(lastDay)
											.stn(stn)
											.dataSize(list.size())
											.build());
					
					
					lastDay = lastDay.plusDays(1);
				}
			}	
*/			
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
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
