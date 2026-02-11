package kr.kro.prjectwwtp.service;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TimeZone;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import jakarta.annotation.PostConstruct;
import kr.kro.prjectwwtp.domain.FakeDate;
import kr.kro.prjectwwtp.domain.FlowSummary;
import kr.kro.prjectwwtp.domain.TmsImputate;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.domain.TmsPredict;
import kr.kro.prjectwwtp.domain.TmsSummary;
import kr.kro.prjectwwtp.persistence.FakeDateRepository;
import kr.kro.prjectwwtp.persistence.FlowSummaryRepository;
import kr.kro.prjectwwtp.persistence.TmsImputateRepository;
import kr.kro.prjectwwtp.persistence.TmsInsertRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import kr.kro.prjectwwtp.persistence.TmsPredictRepository;
import kr.kro.prjectwwtp.persistence.TmsSummaryRepository;
import kr.kro.prjectwwtp.service.TmsImputateService.ImputationConfig;
import kr.kro.prjectwwtp.service.TmsImputateService.OutlierConfig;
import kr.kro.prjectwwtp.util.Util;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsService {
	private final TmsOriginRepository tmsOriginRepo;
	private final TmsImputateRepository tmsImputateRepo;
	private final TmsInsertRepository tmsInsertRepo;
	private final TmsSummaryRepository tmsSummaryRepo;
	private final TmsPredictRepository tmsPredictRepo;
	private final FlowSummaryRepository flowSummaryRepo;
	private final FakeDateRepository fakeDateRepo;
	
	@PostConstruct
	public void init() {
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Seoul"));
	}

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
				LocalDateTime tmsTime = Util.parseDateTime(cols[0]);
				Double toc = Util.parseDoubleOrNullEmptyOk(cols[1]);
				Double ph = Util.parseDoubleOrNullEmptyOk(cols[2]);
				Double ss = Util.parseDoubleOrNullEmptyOk(cols[3]);
				Double flux = Util.parseDoubleOrNullEmptyOk(cols[4]);
				Double tn = Util.parseDoubleOrNullEmptyOk(cols[5]);
				Double tp = Util.parseDoubleOrNullEmptyOk(cols[6]);
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
		tmsInsertRepo.TmsOriginInsert(list);
		return ret;
	}
	
	public List<TmsOrigin> getTmsOriginListByDate(String dateStr) {
		LocalDateTime start = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay();
		LocalDateTime end = LocalDateTime.of(start.getYear(), start.getMonth(), start.getDayOfMonth(), 23, 59, 59);
		List<TmsOrigin> list = tmsOriginRepo.findByTmsTimeBetween(start, end);
		System.out.println("getTmsOriginListByDate size : " + list.size());
		return list;
	}
	
	public boolean existsByTmsTime(LocalDateTime tmsTime) {
		return tmsImputateRepo.existsByTmsTime(tmsTime);
	}
	
	public List<TmsImputate> getTmsImputateListByDate(LocalDateTime end) {
		LocalDateTime start = end.minusDays(1).plusMinutes(1);
		List<TmsImputate> list = tmsImputateRepo.findByTmsTimeBetweenOrderByTmsTime(start, end);
		for(TmsImputate tms : list) {
			String time = tms.getTmsTime().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
			tms.setStrtime(time);
		}
		//System.out.println("start : " + start.toString());
		//System.out.println("end : " + end.toString());
		//System.out.println("getTmsImputateListByDate size : " + list.size());
		return list;
	}
	
	public List<TmsImputate> imputate(LocalDateTime today) {
		LocalDateTime start = today.withHour(0).withMinute(0);
		LocalDateTime end = today.withHour(23).withMinute(59);
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
			t.setFlux(Double.isNaN(flux[i]) ? null : flux[i]);
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
			File file = Util.resolveFilePath(filePath);
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
					sb.append(Util.formatDateTime(tms.getTmsTime())).append(",");
					sb.append(Util.formatDouble(tms.getToc())).append(",");
					sb.append(Util.formatDouble(tms.getPh())).append(",");
					sb.append(Util.formatDouble(tms.getSs())).append(",");
					sb.append(Util.formatDouble(tms.getFlux())).append(",");
					sb.append(Util.formatDouble(tms.getTn())).append(",");
					sb.append(Util.formatDouble(tms.getTp()));
					
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
			File file = Util.resolveFilePath(filePath);
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
						Long tmsNo = Util.parseLongOrNull(cols[0]);
						LocalDateTime tmsTime = Util.parseDateTime(cols[1]);
						Double toc = Util.parseDoubleOrNullEmptyOk(cols[2]);
						Double ph = Util.parseDoubleOrNullEmptyOk(cols[3]);
						Double ss = Util.parseDoubleOrNullEmptyOk(cols[4]);
						Double flux = Util.parseDoubleOrNullEmptyOk(cols[5]);
						Double tn = Util.parseDoubleOrNullEmptyOk(cols[6]);
						Double tp = Util.parseDoubleOrNullEmptyOk(cols[7]);
						
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
	
	public void saveTmsImputateList(List<TmsImputate> list) {
		if(list == null || list.size() == 0) return;
		List<TmsImputate> addList = new ArrayList<>();
		for(TmsImputate tms : list) {
			if(!tmsImputateRepo.existsByTmsTime(tms.getTmsTime())) {
				addList.add(tms);
			}
		}
		tmsInsertRepo.TmsImputateInsert(addList);
	}

	int checkNum = 2600;
	public List<Date> getFakeTmsDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<TmsSummary> summaries = tmsSummaryRepo.findAll();
		
		TmsSummary pre = null;
		for(TmsSummary summary : summaries) {
			if(pre == null) {
				pre = summary;
				continue;
			}
			if( pre.getCount() + summary.getCount() >= checkNum &&
					ChronoUnit.DAYS.between(pre.getTime().toInstant(), summary.getTime().toInstant()) == 1) {
				// 하루전 날짜와의 합계가 checkNum 이상인 경우
				retList.add(summary.getTime());
				}
			pre = summary;
		}
		return retList;
	}
	
	public List<Date> getFakeFlowDatesList() {
		List<Date> retList = new ArrayList<Date>();
		List<FlowSummary> summaries = flowSummaryRepo.findAll();
		
		FlowSummary pre = null;
		for(FlowSummary summary : summaries) {
			if(pre == null) {
				pre = summary;
				continue;
			}
			if( pre.getCount() + summary.getCount() >= checkNum &&
					ChronoUnit.DAYS.between(pre.getTime().toInstant(), summary.getTime().toInstant()) == 1) {
				// 하루전 날짜와의 합계가 checkNum 이상인 경우
				retList.add(summary.getTime());
				}
			pre = summary;
		}
		return retList;
	}
	
	public LocalDateTime getFakeNow() {
		FakeDate fakeDate = fakeDateRepo.findFirstByOrderByTodayDesc();
		// 등록된 값이 오늘 생성한 날짜면 그냥 사용
		if(fakeDate != null
				&& fakeDate.getToday().isAfter(LocalDateTime.now().withHour(0).withMinute(0))) {
			System.out.println("fakeDate.getFakeDate() : " + fakeDate.getTmsDate());
			return fakeDate.getTmsDate();
		}
		
		List<Date> fakeDates = getFakeTmsDatesList();
		Random rand = new Random();
		int idx = rand.nextInt(fakeDates.size());
		Date retDate = fakeDates.get(idx);
		LocalDateTime tmsTime = retDate.toInstant() 
								.atZone(ZoneId.systemDefault())
								.toLocalDateTime();
		
		fakeDates = getFakeFlowDatesList();
		idx = rand.nextInt(fakeDates.size());
		retDate = fakeDates.get(idx);
		LocalDateTime flowTime = retDate.toInstant() 
				.atZone(ZoneId.systemDefault())
				.toLocalDateTime();
		
		fakeDateRepo.save(FakeDate.builder()
				.today(LocalDateTime.now())
				.tmsDate(tmsTime)
				.flowDate(flowTime)
				.build());
		System.out.println("new tmsDate : " + tmsTime + ", " + flowTime);
		return tmsTime;
				
	}
	
//	public List<TmsPredict> findPredictList(LocalDateTime now, LocalDateTime end) {
//		List<TmsPredict> allList = tmsPredictRepo.findByTmsTimeBetweenOrderByTmsTimeAscTmsNoDesc(now, end);
//		// 중복값 제거
//		Map<LocalDateTime, TmsPredict> uniqueMap = new HashMap<>();
//		for(TmsPredict predict : allList) {
//			LocalDateTime tmsTime = predict.getTmsTime().withSecond(0).withNano(0);
//			if(!uniqueMap.containsKey(tmsTime)) {
//				uniqueMap.put(tmsTime, predict);
//			}
//		}
//		List<TmsPredict> result = new ArrayList<>(uniqueMap.values());
//		result.sort((a, b) -> a.getTmsTime().compareTo(b.getTmsTime()));
//		return result;
//	}
	
	public List<TmsPredict> findPredictList(LocalDateTime now, LocalDateTime end) {
		LocalDateTime start = now.withHour(0).withMinute(0).withSecond(0).withNano(0);
		List<TmsPredict> allList = tmsPredictRepo.findByTmsTimeBetweenOrderByTmsTimeAscTmsNoDesc(start, end);
		// 중복값 제거
		Map<LocalDateTime, TmsPredict> uniqueMap = new HashMap<>();
		for(TmsPredict predict : allList) {
			LocalDateTime tmsTime = predict.getTmsTime().withSecond(0).withNano(0);
			if(!uniqueMap.containsKey(tmsTime)) {
				uniqueMap.put(tmsTime, predict);
			}
		}
		List<TmsPredict> result = new ArrayList<>(uniqueMap.values());
		result.sort((a, b) -> a.getTmsTime().compareTo(b.getTmsTime()));
		
		// 누적값으로 전환
		double accFlux = 0;
		for(TmsPredict predict : result) {
			if(predict.getTmsTime().getDayOfMonth() != start.getDayOfMonth())
				accFlux = 0;
			accFlux += predict.getFlux();
			predict.setFlux(accFlux);
		}
		
		// now 보다 이전은 제거
		System.out.println("필터 전 : " + result.size());
		result = result.stream().filter(p -> p.getTmsTime().isAfter(now)).collect(Collectors.toList());
		System.out.println("필터 후 : " + result.size());
		return result;
	}
	
	public void savePredictList(TmsPredict[] array) {
		tmsPredictRepo.saveAll(Arrays.asList(array));
	}
}