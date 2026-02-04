package kr.kro.prjectwwtp.service;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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

import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsLog;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.persistence.TmsImputateRepository;
import kr.kro.prjectwwtp.persistence.TmsLogRepository;
import kr.kro.prjectwwtp.persistence.TmsInsertRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import kr.kro.prjectwwtp.service.TmsImputateService.ImputationConfig;
import kr.kro.prjectwwtp.service.TmsImputateService.OutlierConfig;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsOriginService {

	private final TmsOriginRepository tmsOriginRepo;
	private final TmsImputateRepository tmsImputateRepo;
	private final TmsLogRepository logRepo;
	private final TmsInsertRepository insertRepo;

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

	public static Long parseLongOrNull(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null")) return null;
		try {
			return Long.parseLong(t);
		} catch (NumberFormatException e) {
			return null;
		}
	}

	public static LocalDateTime parseDateTime(String s) {
		String str = s.trim();
		// try several common patterns
		String[] patterns = new String[] {
			"yyyy-MM-dd HH:mm:ss",
			"M/d/yy H:mm",
			"yyyy-MM-dd'T'HH:mm:ss",
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
		System.out.println("getTmsOriginListByDate size : " + list.size());
		return list;
	}
	
	public List<TmsImputate> getTmsImputateListByDate(LocalDateTime end) {
		LocalDateTime start = end.minusDays(1).minusMinutes(1);
		List<TmsImputate> list = tmsImputateRepo.findByTmsTimeBetween(start, end);
		System.out.println("getTmsImputateListByDate size : " + list.size());
		return list;
	}
	
	public List<TmsImputate> imputate(LocalDateTime end) {
		LocalDateTime start = end.minusDays(1).minusMinutes(1);
		List<TmsOrigin> origin = tmsOriginRepo.findByTmsTimeBetween(start, end);
		
		System.out.println("[imputate] origin size=" + origin.size());
		
		// 1분 단위로 1440개의 데이터를 가진 Map 생성
		Map<LocalDateTime, TmsOrigin> dataMap = new HashMap<>();
		for (TmsOrigin tms : origin) {
			dataMap.put(tms.getTmsTime(), tms);
		}
		
		// 시간 초기화
		List<LocalDateTime> times = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			times.add(start.plusMinutes(i));
		}
		
		// 데이터 NaN으로 초기화
		double[] toc = new double[1440];
		double[] ph = new double[1440];
		double[] ss = new double[1440];
		double[] flux = new double[1440];
		double[] tn = new double[1440];
		double[] tp = new double[1440];
		
		Arrays.fill(toc, Double.NaN);
		Arrays.fill(ph, Double.NaN);
		Arrays.fill(ss, Double.NaN);
		Arrays.fill(flux, Double.NaN);
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
		
		System.out.println("[imputate] origin 데이터로 1440개의 배열 생성");
		
		// 2) 결측치 보간
		ImputationConfig impConfig = new ImputationConfig();
		toc = TmsImputateService.imputeMissingWithStrategy(toc, impConfig);
		ph = TmsImputateService.imputeMissingWithStrategy(ph, impConfig);
		ss = TmsImputateService.imputeMissingWithStrategy(ss, impConfig);
		flux = TmsImputateService.imputeMissingWithStrategy(flux, impConfig);
		tn = TmsImputateService.imputeMissingWithStrategy(tn, impConfig);
		tp = TmsImputateService.imputeMissingWithStrategy(tp, impConfig);
		
		System.out.println("[imputate] 데이터 별로 결측치 보간");
		
		// 3) 이상치 탐지 및 처리
		OutlierConfig outConfig = new OutlierConfig();
		toc = TmsImputateService.detectAndHandleOutliers(toc, "toc", outConfig);
		ph = TmsImputateService.detectAndHandleOutliers(ph, "ph", outConfig);
		ss = TmsImputateService.detectAndHandleOutliers(ss, "ss", outConfig);
		flux = TmsImputateService.detectAndHandleOutliers(flux, "flux", outConfig);
		tn = TmsImputateService.detectAndHandleOutliers(tn, "tn", outConfig);
		tp = TmsImputateService.detectAndHandleOutliers(tp, "tp", outConfig);
		
		System.out.println("[imputate] 이상치 처리");
		
		// 4) List<TmsImputate>으로 변환
		List<TmsImputate> result = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			TmsImputate t = new TmsImputate();
			t.setTmsTime(times.get(i));
			t.setToc(Double.isNaN(toc[i]) ? null : toc[i]);
			t.setPh(Double.isNaN(ph[i]) ? null : ph[i]);
			t.setSs(Double.isNaN(ss[i]) ? null : ss[i]);
			Double fluxValue = Double.isNaN(flux[i]) ? null : flux[i];
			t.setFlux((int)fluxValue.doubleValue());
			t.setTn(Double.isNaN(tn[i]) ? null : tn[i]);
			t.setTp(Double.isNaN(tp[i]) ? null : tp[i]);
			result.add(t);
		}
		
		System.out.println("[imputate] 데이터 구성=" + result.size());
		checkNullValues(result);
		return result;
	}
	
	/**
	 * TmsOrigin 리스트의 NULL 값 분석
	 * 각 필드별 NULL 개수와 비율을 출력
	 * 
	 * @param list TmsOrigin 리스트
	 */
	private void checkNullValues(List<TmsImputate> list) {
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
		for (TmsImputate tms : list) {
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
	
	/**
	 * TmsOrigin 리스트를 CSV 파일로 저장
	 * 
	 * @param list TmsOrigin 리스트
	 * @param filePath 저장할 파일 경로 (상대 경로 또는 절대 경로)
	 * @return 저장 성공 여부
	 * @throws Exception 파일 저장 중 발생하는 예외
	 */
	public boolean saveToCsv(List<TmsImputate> list, String filePath) throws Exception {
		if (list == null || list.isEmpty()) {
			System.out.println("[saveToCsv] 저장할 데이터가 없습니다");
			return false;
		}
		
		try {
			// 파일 경로 해석 (홈 디렉토리, 상대 경로, 절대 경로 모두 지원)
			File file = resolveFilePath(filePath);
			System.out.println("[saveToCsv] 해석된 파일 경로: " + file.getAbsolutePath());
			
			// 파일 경로의 디렉토리 생성
			File parentDir = file.getParentFile();
			if (parentDir != null && !parentDir.exists()) {
				boolean dirCreated = parentDir.mkdirs();
				if (!dirCreated && !parentDir.exists()) {
					throw new Exception("디렉토리 생성 실패: " + parentDir.getAbsolutePath());
				}
				System.out.println("[saveToCsv] 디렉토리 생성: " + parentDir.getAbsolutePath());
			}
			
			// 부모 디렉토리 쓰기 권한 확인
			if (parentDir != null && !parentDir.canWrite()) {
				throw new Exception("디렉토리 쓰기 권한 없음: " + parentDir.getAbsolutePath());
			}
			
			// UTF-8 인코딩으로 CSV 파일 작성
			try (BufferedWriter bw = new BufferedWriter(
					new OutputStreamWriter(new FileOutputStream(file.getAbsolutePath()), "UTF-8"))) {
				
				// 헤더 작성
				//bw.write("tmsNo,tmsTime,toc,ph,ss,flux,tn,tp");
				bw.write("SYS_TIME,TOC_VU,PH_VU,SS_VU,FLUX_VU,TN_VU,TP_VU");
				bw.newLine();
				
				// 데이터 작성
				for (TmsImputate tms : list) {
					StringBuilder sb = new StringBuilder();
					//sb.append(tms.getTmsNo()).append(",");
					sb.append(formatDateTime(tms.getTmsTime())).append(",");
					sb.append(formatDouble(tms.getToc())).append(",");
					sb.append(formatDouble(tms.getPh())).append(",");
					sb.append(formatDouble(tms.getSs())).append(",");
					sb.append(formatInteger(tms.getFlux())).append(",");
					sb.append(formatDouble(tms.getTn())).append(",");
					sb.append(formatDouble(tms.getTp()));
					
					bw.write(sb.toString());
					bw.newLine();
				}
			}
			
			System.out.println("[saveToCsv] CSV 파일 저장 성공: " + filePath);
			System.out.println("[saveToCsv] 저장된 행 수: " + list.size());
			return true;
			
		} catch (Exception e) {
			System.err.println("[saveToCsv] CSV 파일 저장 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			throw new Exception("CSV 파일 저장 중 오류가 발생했습니다: " + e.getMessage());
		}
	}
	
	/**
	 * CSV 파일로부터 TmsOrigin 리스트를 로드
	 * 
	 * @param filePath 로드할 파일 경로 (상대 경로 또는 절대 경로)
	 * @return 로드된 TmsOrigin 리스트
	 * @throws Exception 파일 로드 중 발생하는 예외
	 */
	public List<TmsImputate> loadFromCsv(String filePath) {
		List<TmsImputate> list = new ArrayList<>();
		
		try {
			// 파일 경로 해석 (홈 디렉토리, 상대 경로, 절대 경로 모두 지원)
			File file = resolveFilePath(filePath);
			System.out.println("[loadFromCsv] 해석된 파일 경로: " + file.getAbsolutePath());
			
			// 파일 존재 여부 확인
			if (!file.exists()) {
				//throw new Exception("파일을 찾을 수 없음: " + file.getAbsolutePath());
				return null;
			}
			
			// 파일 읽기 권한 확인
			if (!file.canRead()) {
				//throw new Exception("파일 읽기 권한 없음: " + file.getAbsolutePath());
				return null;
			}
			
			// UTF-8 인코딩으로 CSV 파일 읽기
			try (BufferedReader br = new BufferedReader(
					new InputStreamReader(new java.io.FileInputStream(file.getAbsolutePath()), "UTF-8"))) {
				
				String line;
				int lineNo = 0;
				
				while ((line = br.readLine()) != null) {
					lineNo++;
					
					// 빈 라인 스킵
					if (line.isEmpty()) {
						continue;
					}
					
					// 헤더 라인 스킵
					if (lineNo == 1 && line.contains("tmsNo")) {
						continue;
					}
					
					// CSV 파싱
					String[] cols = line.split(",");
					if (cols.length < 8) {
						System.out.println("[loadFromCsv] 경고: 라인 " + lineNo + "의 컬럼 수가 부족합니다. 스킵");
						continue;
					}
					
					try {
						Long tmsNo = parseLongOrNull(cols[0]);
						LocalDateTime tmsTime = parseDateTime(cols[1]);
						Double toc = parseDoubleOrNullEmptyOk(cols[2]);
						Double ph = parseDoubleOrNullEmptyOk(cols[3]);
						Double ss = parseDoubleOrNullEmptyOk(cols[4]);
						Integer flux = parseIntOrNullEmptyOk(cols[5]);
						Double tn = parseDoubleOrNullEmptyOk(cols[6]);
						Double tp = parseDoubleOrNullEmptyOk(cols[7]);
						
						TmsImputate tms = TmsImputate.builder()
							.tmsNo(tmsNo)
							.tmsTime(tmsTime)
							.toc(toc)
							.ph(ph)
							.ss(ss)
							.flux(flux)
							.tn(tn)
							.tp(tp)
							.build();
						
						list.add(tms);
						
					} catch (Exception e) {
						System.out.println("[loadFromCsv] 경고: 라인 " + lineNo + " 파싱 중 오류 발생 - " + e.getMessage() + ", 스킵");
						continue;
					}
				}
			}
			
			System.out.println("[loadFromCsv] CSV 파일 로드 성공: " + filePath);
			System.out.println("[loadFromCsv] 로드된 행 수: " + list.size());
			return list;
			
		} catch (Exception e) {
			System.err.println("[loadFromCsv] CSV 파일 로드 중 오류 발생: " + e.getMessage());
			e.printStackTrace();
			//throw new Exception("CSV 파일 로드 중 오류가 발생했습니다: " + e.getMessage());
			return null;
		}
	}
	
	/**
	 * LocalDateTime을 문자열로 포맷
	 * 
	 * @param dateTime LocalDateTime 객체
	 * @return 포맷된 문자열 (yyyy-MM-dd HH:mm:ss)
	 */
	private String formatDateTime(LocalDateTime dateTime) {
		if (dateTime == null) {
			return "";
		}
		return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
	}
	
	/**
	 * Double 값을 문자열로 포맷
	 * 
	 * @param value Double 값
	 * @return 포맷된 문자열 (null이면 빈 문자열)
	 */
	private String formatDouble(Double value) {
		if (value == null || value.isNaN()) {
			return "";
		}
		return String.valueOf(value);
	}
	
	/**
	 * Integer 값을 문자열로 포맷
	 * 
	 * @param value Integer 값
	 * @return 포맷된 문자열 (null이면 빈 문자열)
	 */
	private String formatInteger(Integer value) {
		if (value == null) {
			return "";
		}
		return String.valueOf(value);
	}
	
	/**
	 * 파일 경로를 정규화하여 절대 경로로 변환
	 * ~/Downloads/ 형태의 홈 디렉토리 경로도 처리
	 * 
	 * @param filePath 파일 경로 (상대/절대/홈 디렉토리 경로)
	 * @return 절대 경로 File 객체
	 */
	private File resolveFilePath(String filePath) {
		File file = new File(filePath);
		
		// 이미 절대 경로인 경우
		if (file.isAbsolute()) {
			return file;
		}
		
		// 홈 디렉토리 경로 처리 (~/ 또는 ~\)
		if (filePath.startsWith("~" + File.separator) || filePath.startsWith("~/")) {
			String userHome = System.getProperty("user.home");
			String relativePath = filePath.substring(2); // ~/ 제거
			return new File(userHome, relativePath);
		}
		
		// 상대 경로인 경우 현재 작업 디렉토리 기준
		String workingDir = System.getProperty("user.dir");
		return new File(workingDir, filePath);
	}
	
	public void saveTmsImputateList(List<TmsImputate> list) {
		if(list == null || list.size() == 0) return;
		List<TmsImputate> addList = new ArrayList<>();
		for(TmsImputate tms : list) {
			if(!tmsImputateRepo.existsByTmsTime(tms.getTmsTime())) {
				addList.add(tms);
			}
		}
		insertRepo.TmsImputateInsert(addList);
	}
}