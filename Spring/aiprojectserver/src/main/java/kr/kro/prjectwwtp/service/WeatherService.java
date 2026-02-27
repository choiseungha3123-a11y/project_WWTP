package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import kr.kro.prjectwwtp.domain.Weather;
import kr.kro.prjectwwtp.domain.WeatherDTO;
import kr.kro.prjectwwtp.domain.WeatherSummary;
import kr.kro.prjectwwtp.domain.WeatherSummaryId;
import kr.kro.prjectwwtp.persistence.WeatherRepository;
import kr.kro.prjectwwtp.persistence.WeatherSummaryRepository;
import kr.kro.prjectwwtp.util.ImputateUtil;
import kr.kro.prjectwwtp.util.ImputateUtil.ImputationConfig;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class WeatherService {
	private final WeatherRepository weatherRepo;
	private final WeatherSummaryRepository summaryRepo;
	
	public Weather findById(long id) {
		Optional<Weather> opt = weatherRepo.findById(id);
		if(opt.isEmpty())
			return null;
		return opt.get();
	}
	
	public Weather findFirstByStnOrderByLogTimeDesc(int stn) {
		return weatherRepo.findFirstByStnOrderByLogTimeDesc(stn);
	}
	
	public void saveWeatherList(List<Weather> list) {
		weatherRepo.saveAll(list);
	}
	
	public void deleteWeatherList(List<Weather> list) {
		weatherRepo.deleteAll(list);
	}
	
	public List<Weather> findByLogTimeBetween(LocalDateTime start, LocalDateTime end) {
		return weatherRepo.findByLogTimeBetween(start, end);
	}
	
	public List<WeatherDTO> findWeatherDTOByLogTimeBetween(LocalDateTime start, LocalDateTime end) {
		List<Weather> list = weatherRepo.findByLogTimeBetween(start, end);
		List<WeatherDTO> ret = new ArrayList<>();
		for(Weather w : list)
			ret.add(new WeatherDTO(w));
		return ret;
	}
	
	public List<Weather> findByStnAndLogTimeBetween(int stn, LocalDateTime start, LocalDateTime end) {
		return weatherRepo.findByStnAndLogTimeBetween(stn, start, end);
	}
	
	public List<WeatherDTO> findWeatherDTOByStnAndLogTimeBetween(int stn, LocalDateTime start, LocalDateTime end) {
		List<Weather> list = weatherRepo.findByStnAndLogTimeBetween(stn, start, end); 
		List<WeatherDTO> ret = new ArrayList<>();
		for(Weather w : list) {
			if(!w.isValid()) continue;
			ret.add(new WeatherDTO(w));
		}
		if(ret.size() < 1440) {
			ret = imputate(start, ret);
		}
		return ret;
	}
	
	public int getWeatherCountByStnAndTimeBetween(int stn, LocalDateTime date) {
		int cnt = 0;
		Optional<WeatherSummary> opt = summaryRepo.findById(WeatherSummaryId.builder().stn(stn).time(date.minusDays(1)).build());
		if(opt.isPresent())
			cnt += opt.get().getCount();
		opt = summaryRepo.findById(WeatherSummaryId.builder().stn(stn).time(date).build());
		if(opt.isPresent())
			cnt += opt.get().getCount();
		return cnt;
	}
	
	private List<WeatherDTO> imputate(LocalDateTime start, List<WeatherDTO> origin) {
		System.out.println("[imputate] origin size=" + origin.size());
		
		// 1분 단위로 1440개의 데이터를 가진 Map 생성
		Map<LocalDateTime, WeatherDTO> dataMap = new HashMap<>();
		for (WeatherDTO dto : origin) {
			dataMap.put(LocalDateTime.parse(dto.getTime(), DateTimeFormatter.ofPattern("yyyyMMddHHmmss")), dto);
		}
		
		// 시간 초기화
		List<LocalDateTime> times = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			times.add(start.plusMinutes(i));
		}
		
		// 데이터 NaN으로 초기화
		double distance = origin.get(0).getDistance();
		double[] ta = new double[1440];
		double[] rn15m = new double[1440];
		double[] rn60m = new double[1440];
		double[] rn12h = new double[1440];
		double[] rnday = new double[1440];
		double[] hm = new double[1440];
		double[] td = new double[1440];
		
		Arrays.fill(ta, Double.NaN);
		Arrays.fill(rn15m, Double.NaN);
		Arrays.fill(rn60m, Double.NaN);
		Arrays.fill(rn12h, Double.NaN);
		Arrays.fill(rnday, Double.NaN);
		Arrays.fill(hm, Double.NaN);
		Arrays.fill(td, Double.NaN);
		
		// origin 데이터 채우기
		for (int i = 0; i < 1440; i++) {
			LocalDateTime t = times.get(i);
			WeatherDTO orig = dataMap.get(t);
			if (orig != null) {
				if (orig.getTa() != null) ta[i] = orig.getTa();
				if (orig.getRn15m() != null) rn15m[i] = orig.getRn15m();
				if (orig.getRn60m() != null) rn60m[i] = orig.getRn60m();
				if (orig.getRn12h() != null) rn12h[i] = orig.getRn12h();
				if (orig.getRnday() != null) rnday[i] = orig.getRnday();
				if (orig.getHm() != null) hm[i] = orig.getHm();
				if (orig.getTd() != null) td[i] = orig.getTd();
			}
		}
		
		System.out.println("[imputate] origin 데이터로 1440개의 배열 생성");
		
		// 2) 결측치 보간
		ImputationConfig impConfig = new ImputationConfig();
		ta = ImputateUtil.imputeMissingWithStrategy(ta, impConfig);
		rn15m = ImputateUtil.imputeMissingWithStrategy(rn15m, impConfig);
		rn60m = ImputateUtil.imputeMissingWithStrategy(rn60m, impConfig);
		rn12h = ImputateUtil.imputeMissingWithStrategy(rn12h, impConfig);
		rnday = ImputateUtil.imputeMissingWithStrategy(rnday, impConfig);
		hm = ImputateUtil.imputeMissingWithStrategy(hm, impConfig);
		td = ImputateUtil.imputeMissingWithStrategy(td, impConfig);
		
		System.out.println("[imputate] 데이터 별로 결측치 보간");
		
		// 4) List<FlowImputate>으로 변환
		List<WeatherDTO> result = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			WeatherDTO t = new WeatherDTO();
			t.setTime(times.get(i).format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")));
			t.setTa(Double.isNaN(ta[i]) ? null : ta[i]);
			t.setRn15m(Double.isNaN(rn15m[i]) ? null : rn15m[i]);
			t.setRn60m(Double.isNaN(rn60m[i]) ? null : rn60m[i]);
			t.setRn12h(Double.isNaN(rn12h[i]) ? null : rn12h[i]);
			t.setRnday(Double.isNaN(rnday[i]) ? null : rnday[i]);
			t.setHm(Double.isNaN(hm[i]) ? null : hm[i]);
			t.setTd(Double.isNaN(td[i]) ? null : td[i]);
			t.setDistance(distance);
			result.add(t);
		}
		
		System.out.println("[imputate] 데이터 구성=" + result.size());
		checkNullValues(result);
		return result;
	}
	private void checkNullValues(List<WeatherDTO> list) {
		if (list == null || list.isEmpty()) {
			System.out.println("[NULL Check] 리스트가 비어있습니다");
			return;
		}
		
		int totalRows = list.size();
		int taNullCount = 0;
		int rn15mNullCount = 0;
		int rn60mNullCount = 0;
		int rn12hNullCount = 0;
		int rndayNullCount = 0;
		int hmNullCount = 0;
		int tdNullCount = 0;
		
		// NULL 값 개수 계산
		for (WeatherDTO dto : list) {
			if (dto.getTa() == null) taNullCount++;
			if (dto.getRn15m() == null) rn15mNullCount++;
			if (dto.getRn60m() == null) rn60mNullCount++;
			if (dto.getRn12h() == null) rn12hNullCount++;
			if (dto.getRnday() == null) rndayNullCount++;
			if (dto.getHm() == null) hmNullCount++;
			if (dto.getTd() == null) tdNullCount++;
		}
		
		// 결과 출력
		System.out.println("=== WeatherOrigin NULL 값 분석 ===");
		System.out.println("총 행 수: " + totalRows);
		System.out.println();
		System.out.printf("ta   - NULL: %4d / %4d (%.2f%%)%n", taNullCount, totalRows, (double) taNullCount / totalRows * 100);
		System.out.printf("rn15m    - NULL: %4d / %4d (%.2f%%)%n", rn15mNullCount, totalRows, (double) rn15mNullCount / totalRows * 100);
		System.out.printf("rn60m    - NULL: %4d / %4d (%.2f%%)%n", rn60mNullCount, totalRows, (double) rn60mNullCount / totalRows * 100);
		System.out.printf("rn12h  - NULL: %4d / %4d (%.2f%%)%n", rn12hNullCount, totalRows, (double) rn12hNullCount / totalRows * 100);
		System.out.printf("rnday  - NULL: %4d / %4d (%.2f%%)%n", rndayNullCount, totalRows, (double) rndayNullCount / totalRows * 100);
		System.out.printf("hm  - NULL: %4d / %4d (%.2f%%)%n", hmNullCount, totalRows, (double) hmNullCount / totalRows * 100);
		System.out.printf("td  - NULL: %4d / %4d (%.2f%%)%n", tdNullCount, totalRows, (double) tdNullCount / totalRows * 100);
		System.out.println();
		
		// 전체 NULL 개수
		int totalNulls = taNullCount + rn15mNullCount + rn60mNullCount + rn12hNullCount + rndayNullCount + hmNullCount + tdNullCount;
		int totalFields = totalRows * 7;
		System.out.printf("전체 NULL: %d / %d (%.2f%%)%n", totalNulls, totalFields, (double) totalNulls / totalFields * 100);
		System.out.println("==============================");
	}
	
	public void modifyWeahter(Weather data, double ta, double rn15m, double rn60m, double rn12h, double rnday, double hm, double td) {
		data.setTa(ta);
		data.setRn15m(rn15m);
		data.setRn60m(rn60m);
		data.setRn12h(rn12h);
		data.setRnday(rnday);
		data.setHm(hm);
		data.setTd(td);
		weatherRepo.save(data);
	}

}
