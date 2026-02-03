package kr.kro.prjectwwtp.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import kr.kro.prjectwwtp.domain.TmsLog;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.imputation.TmsDataProcessor;
import kr.kro.prjectwwtp.imputation.TmsDataProcessor.ImputationConfig;
import kr.kro.prjectwwtp.imputation.TmsDataProcessor.OutlierConfig;
import kr.kro.prjectwwtp.persistence.TmsLogRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginInsertRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsOriginService {

	private final TmsOriginRepository tmsOriginRepo;
	private final TmsLogRepository logRepo;
	private final TmsOriginInsertRepository insertRepo;

	/**
	 * Parse CSV file and save TmsOrigin entries.
	 * Returns detailed import statistics in TmsImportResult.
	 */
	@Transactional
	public int saveFromCsv(MultipartFile file) throws Exception {
		if (file == null || file.isEmpty()) return 0;
		int batchSize = 3000;

		int addCount = 0;
		int lineNo = 0;
		List<TmsOrigin> list = new ArrayList<>();
		String line;
		try (BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream(), "UTF-8"))) {
			while ((line = br.readLine()) != null) {
				lineNo++;
				if(line.isEmpty()) {
					continue;
				}
				// 첫 라인이 컬럼이면 skip
				if(lineNo == 1 && (line.contains("SYS_TIME") || line.contains("TOC_VU") || line.contains("PH_VU"))) {
					continue;
				}
				String[] cols = line.split(",");
				// 데이터가 모자를 때 skip 유찬씨랑 상의해서 수정 
				if(cols.length < 7) {
					continue; // 
				}
				LocalDateTime tmsTime = parseDateTime(cols[0]);
				Double toc = parseDoubleOrNullEmptyOk(cols[1]);
				Double ph = parseDoubleOrNullEmptyOk(cols[2]);
				Double ss = parseDoubleOrNullEmptyOk(cols[3]);
				Integer flux = parseIntOrNullEmptyOk(cols[4]);
				Double tn = parseDoubleOrNullEmptyOk(cols[5]);
				Double tp = parseDoubleOrNullEmptyOk(cols[6]);
				TmsOrigin t = TmsOrigin.builder()
					.tmsTime(tmsTime)
					.toc(toc)
					.ph(ph)
					.ss(ss)
					.flux(flux)
					.tn(tn)
					.tp(tp)
					.build();
				list.add(t);		
				
				if(list.size() >= batchSize) {
					addCount += saveBatch(list);
					System.out.println("list Count : " + list.size());
					System.out.println("addCount: " + addCount);
					list.clear();
				}
			}
			if(list.size() >= 0) {
				addCount += saveBatch(list);
				System.out.println("list Count : " + list.size());
				System.out.println("addCount: " + addCount);
				list.clear();
			}
			logRepo.save(TmsLog.builder()
				.type("upload")
				.count(list.size())
				.build());
			
			System.out.println("lineNo: " + lineNo);
			System.out.println("Final addCount: " + addCount);
			return addCount;	
		}
	}
	
	public int saveBatch(List<TmsOrigin> list) {
		if(list == null || list.size() == 0) return 0;
		LocalDateTime firstTime = list.get(0).getTmsTime();
		LocalDateTime lastTime = list.get(list.size()-1).getTmsTime();
		List<TmsOrigin> existing = tmsOriginRepo.findByTmsTimeBetween(firstTime, lastTime);
		for(TmsOrigin e : existing) {
			list.removeIf(tms -> tms.getTmsTime().isEqual(e.getTmsTime()));
		}
		int ret = list.size();
		insertRepo.TmsOriginInsert(list);
		return ret;
	}


	// parse helpers that treat empty string as null (explicit)
	public static Double parseDoubleOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null") || t.equalsIgnoreCase("-99.0") || t.equalsIgnoreCase("-99.9")) return null;
		return Double.parseDouble(t);
	}

	public static Integer parseIntOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null")) return null;
		return Integer.parseInt(t);
	}

	public static LocalDateTime parseDateTime(String s) {
		String str = s.trim();
		// try several common patterns
		String[] patterns = new String[] {
			"M/d/yy H:mm",
		};
		for (String p : patterns) {
			try {
				DateTimeFormatter f = DateTimeFormatter.ofPattern(p);
				return LocalDateTime.parse(str, f);
			} catch (Exception e) {
				// try next
			}
		}
		// try ISO parse
		try {
			return LocalDateTime.parse(str);
		} catch (Exception e) {
			throw new IllegalArgumentException("날짜 형식이 올바르지 않습니다: " + s);
		}
	}
	
	public List<TmsOrigin> getTmsOriginListByDate(String dateStr) {
		LocalDateTime start = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay();
		LocalDateTime end = LocalDateTime.of(start.getYear(), start.getMonth(), start.getDayOfMonth(), 23, 59, 59);
		List<TmsOrigin> list = tmsOriginRepo.findByTmsTimeBetween(start, end);
		System.out.println("size : " + list.size());
		return list;
	}
	
	public List<TmsOrigin> imputate(String dateStr) {
		LocalDateTime start = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay();
		LocalDateTime end = LocalDateTime.of(start.getYear(), start.getMonth(), start.getDayOfMonth(), 23, 59, 59);
		List<TmsOrigin> origin = tmsOriginRepo.findByTmsTimeBetween(start, end);
		
		System.out.println("[imputate] origin size=" + origin.size());
		
		// 1분 단위로 1440개 행 생성
		Map<LocalDateTime, TmsOrigin> dataMap = new HashMap<>();
		for (TmsOrigin tms : origin) {
			dataMap.put(tms.getTmsTime(), tms);
		}
		
		List<LocalDateTime> times = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			times.add(start.plusMinutes(i));
		}
		
		// 1440개 행으로 DataFrame 구성 (NaN으로 초기화)
		double[] toc = new double[1440];
		double[] ph = new double[1440];
		double[] ss = new double[1440];
		int[] flux = new int[1440];
		double[] tn = new double[1440];
		double[] tp = new double[1440];
		
		Arrays.fill(toc, Double.NaN);
		Arrays.fill(ph, Double.NaN);
		Arrays.fill(ss, Double.NaN);
		Arrays.fill(flux, Integer.MIN_VALUE);
		Arrays.fill(tn, Double.NaN);
		Arrays.fill(tp, Double.NaN);
		
		// origin 데이터 채우기
		for (int i = 0; i < 1440; i++) {
			LocalDateTime t = times.get(i);
			TmsOrigin orig = dataMap.get(t);
			if (orig != null) {
				if (orig.getToc() != null) toc[i] = orig.getToc();
				if (orig.getPh() != null) ph[i] = orig.getPh();
				if (orig.getSs() != null) ss[i] = orig.getSs();
				if (orig.getFlux() != null) flux[i] = orig.getFlux();
				if (orig.getTn() != null) tn[i] = orig.getTn();
				if (orig.getTp() != null) tp[i] = orig.getTp();
			}
		}
		
		System.out.println("[imputate] after reindex rows=1440");
		
		// 2) 결측치 보간
		ImputationConfig impConfig = new ImputationConfig();
		toc = TmsDataProcessor.imputeMissingWithStrategy(toc, impConfig);
		ph = TmsDataProcessor.imputeMissingWithStrategy(ph, impConfig);
		ss = TmsDataProcessor.imputeMissingWithStrategy(ss, impConfig);
		tn = TmsDataProcessor.imputeMissingWithStrategy(tn, impConfig);
		tp = TmsDataProcessor.imputeMissingWithStrategy(tp, impConfig);
		
		System.out.println("[imputate] after imputation rows=1440");
		
		// 3) 이상치 탐지 및 처리
		OutlierConfig outConfig = new OutlierConfig();
		toc = TmsDataProcessor.detectAndHandleOutliers(toc, "toc", outConfig);
		ph = TmsDataProcessor.detectAndHandleOutliers(ph, "ph", outConfig);
		ss = TmsDataProcessor.detectAndHandleOutliers(ss, "ss", outConfig);
		tn = TmsDataProcessor.detectAndHandleOutliers(tn, "tn", outConfig);
		tp = TmsDataProcessor.detectAndHandleOutliers(tp, "tp", outConfig);
		
		System.out.println("[imputate] after outlier handling rows=1440");
		
		// 4) List<TmsOrigin>으로 변환
		List<TmsOrigin> result = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			TmsOrigin t = new TmsOrigin();
			t.setTmsTime(times.get(i));
			t.setToc(Double.isNaN(toc[i]) ? null : toc[i]);
			t.setPh(Double.isNaN(ph[i]) ? null : ph[i]);
			t.setSs(Double.isNaN(ss[i]) ? null : ss[i]);
			t.setFlux(flux[i] == Integer.MIN_VALUE ? null : flux[i]);
			t.setTn(Double.isNaN(tn[i]) ? null : tn[i]);
			t.setTp(Double.isNaN(tp[i]) ? null : tp[i]);
			result.add(t);
		}
		
		System.out.println("[imputate] final result size=" + result.size());
		checkNullValues(result);
		return result;
	}
	
	/**
	 * TmsOrigin 리스트의 NULL 값 분석
	 * 각 필드별 NULL 개수와 비율을 출력
	 * 
	 * @param list TmsOrigin 리스트
	 */
	private void checkNullValues(List<TmsOrigin> list) {
		if (list == null || list.isEmpty()) {
			System.out.println("[NULL Check] 리스트가 비어있습니다");
			return;
		}
		
		int totalRows = list.size();
		int tocNullCount = 0;
		int phNullCount = 0;
		int ssNullCount = 0;
		int fluxNullCount = 0;
		int tnNullCount = 0;
		int tpNullCount = 0;
		
		// NULL 값 개수 계산
		for (TmsOrigin tms : list) {
			if (tms.getToc() == null) tocNullCount++;
			if (tms.getPh() == null) phNullCount++;
			if (tms.getSs() == null) ssNullCount++;
			if (tms.getFlux() == null) fluxNullCount++;
			if (tms.getTn() == null) tnNullCount++;
			if (tms.getTp() == null) tpNullCount++;
		}
		
		// 결과 출력
		System.out.println("=== TmsOrigin NULL 값 분석 ===");
		System.out.println("총 행 수: " + totalRows);
		System.out.println();
		System.out.printf("toc   - NULL: %4d / %4d (%.2f%%)%n", tocNullCount, totalRows, (double) tocNullCount / totalRows * 100);
		System.out.printf("ph    - NULL: %4d / %4d (%.2f%%)%n", phNullCount, totalRows, (double) phNullCount / totalRows * 100);
		System.out.printf("ss    - NULL: %4d / %4d (%.2f%%)%n", ssNullCount, totalRows, (double) ssNullCount / totalRows * 100);
		System.out.printf("flux  - NULL: %4d / %4d (%.2f%%)%n", fluxNullCount, totalRows, (double) fluxNullCount / totalRows * 100);
		System.out.printf("tn    - NULL: %4d / %4d (%.2f%%)%n", tnNullCount, totalRows, (double) tnNullCount / totalRows * 100);
		System.out.printf("tp    - NULL: %4d / %4d (%.2f%%)%n", tpNullCount, totalRows, (double) tpNullCount / totalRows * 100);
		System.out.println();
		
		// 전체 NULL 개수
		int totalNulls = tocNullCount + phNullCount + ssNullCount + fluxNullCount + tnNullCount + tpNullCount;
		int totalFields = totalRows * 6;
		System.out.printf("전체 NULL: %d / %d (%.2f%%)%n", totalNulls, totalFields, (double) totalNulls / totalFields * 100);
		System.out.println("==============================");
	}
}