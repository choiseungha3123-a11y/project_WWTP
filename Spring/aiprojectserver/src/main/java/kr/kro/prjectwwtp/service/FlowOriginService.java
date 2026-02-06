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

import kr.kro.prjectwwtp.domain.FlowImputate;
import kr.kro.prjectwwtp.domain.FlowLog;
import kr.kro.prjectwwtp.domain.FlowOrigin;
import kr.kro.prjectwwtp.persistence.FlowImputateRepository;
import kr.kro.prjectwwtp.persistence.FlowInsertRepository;
import kr.kro.prjectwwtp.persistence.FlowLogRepository;
import kr.kro.prjectwwtp.persistence.FlowOriginRepository;
import kr.kro.prjectwwtp.service.TmsImputateService.ImputationConfig;
import kr.kro.prjectwwtp.service.TmsImputateService.OutlierConfig;
import kr.kro.prjectwwtp.util.Util;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class FlowOriginService {

	private final FlowOriginRepository flowOriginRepo;
	private final FlowImputateRepository flowImputateRepo;
	private final FlowLogRepository logRepo;
	private final FlowInsertRepository insertRepo;

	/**
	 * Parse CSV file and save FlowOrigin entries.
	 * Returns detailed import statistics in FlowImportResult.
	 */
	@Transactional
	public int saveFromCsv(MultipartFile file) throws Exception {
		if (file == null || file.isEmpty()) return 0;
		int batchSize = 3000;

		int addCount = 0;
		int lineNo = 0;
		List<FlowOrigin> list = new ArrayList<>();
		String line;
		try (BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream(), "UTF-8"))) {
			while ((line = br.readLine()) != null) {
				lineNo++;
				if(line.isEmpty()) {
					continue;
				}
				// 첫 라인이 컬럼이면 skip
				if(lineNo == 1 && (line.contains("data_save_dt") || line.contains("유량조정조A") || line.contains("유량조정조B"))) {
					continue;
				}
				String[] cols = line.split(",");
				// 데이터가 모자를 때 skip 유찬씨랑 상의해서 수정 
				if(cols.length < 5) {
					continue; // 
				}
				LocalDateTime flowTime = Util.parseDateTime(cols[0]);
				Double flowA = Util.parseDoubleOrNullEmptyOk(cols[1]);
				Double flowB = Util.parseDoubleOrNullEmptyOk(cols[2]);
				Double levelA = Util.parseDoubleOrNullEmptyOk(cols[3]);
				Double levelB = Util.parseDoubleOrNullEmptyOk(cols[4]);
				FlowOrigin t = FlowOrigin.builder()
					.flowTime(flowTime)
					.flowA(flowA)
					.flowB(flowB)
					.levelA(levelA)
					.levelB(levelB)
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
			logRepo.save(FlowLog.builder()
				.type("upload")
				.count(list.size())
				.build());
			
			System.out.println("lineNo: " + lineNo);
			System.out.println("Final addCount: " + addCount);
			return addCount;	
		}
	}
	
	public int saveBatch(List<FlowOrigin> list) {
		if(list == null || list.size() == 0) return 0;
		LocalDateTime firstTime = list.get(0).getFlowTime();
		LocalDateTime lastTime = list.get(list.size()-1).getFlowTime();
		List<FlowOrigin> existing = flowOriginRepo.findByFlowTimeBetween(firstTime, lastTime);
		for(FlowOrigin e : existing) {
			list.removeIf(flow -> flow.getFlowTime().isEqual(e.getFlowTime()));
		}
		int ret = list.size();
		insertRepo.FlowOriginInsert(list);
		return ret;
	}
	
	public List<FlowOrigin> getFlowOriginListByDate(String dateStr) {
		LocalDateTime start = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay();
		LocalDateTime end = LocalDateTime.of(start.getYear(), start.getMonth(), start.getDayOfMonth(), 23, 59, 59);
		List<FlowOrigin> list = flowOriginRepo.findByFlowTimeBetween(start, end);
		System.out.println("getFlowOriginListByDate size : " + list.size());
		return list;
	}
	
	public boolean existsByFlowTime(LocalDateTime flowTime) {
		return flowImputateRepo.existsByFlowTime(flowTime);
	}
	
	public List<FlowImputate> getFlowImputateListByDate(LocalDateTime end) {
		LocalDateTime start = end.minusDays(1).plusMinutes(1);
		List<FlowImputate> list = flowImputateRepo.findByFlowTimeBetweenOrderByFlowTime(start, end);
		System.out.println("start : " + start.toString());
		System.out.println("end : " + end.toString());
		System.out.println("getFlowImputateListByDate size : " + list.size());
		return list;
	}
	
	public List<FlowImputate> imputate(LocalDateTime today) {
		LocalDateTime start = today.withHour(0).withMinute(0);
		LocalDateTime end = today.withHour(23).withMinute(59);
		List<FlowOrigin> origin = flowOriginRepo.findByFlowTimeBetween(start, end);
		
		System.out.println("[imputate] origin size=" + origin.size());
		
		// 1분 단위로 1440개의 데이터를 가진 Map 생성
		Map<LocalDateTime, FlowOrigin> dataMap = new HashMap<>();
		for (FlowOrigin flow : origin) {
			dataMap.put(flow.getFlowTime(), flow);
		}
		
		// 시간 초기화
		List<LocalDateTime> times = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			times.add(start.plusMinutes(i));
		}
		
		// 데이터 NaN으로 초기화
		double[] flowA = new double[1440];
		double[] flowB = new double[1440];
		double[] levelA = new double[1440];
		double[] levelB = new double[1440];
		
		Arrays.fill(flowA, Double.NaN);
		Arrays.fill(flowB, Double.NaN);
		Arrays.fill(levelA, Double.NaN);
		Arrays.fill(levelB, Double.NaN);
		
		// origin 데이터 채우기
		for (int i = 0; i < 1440; i++) {
			LocalDateTime t = times.get(i);
			FlowOrigin orig = dataMap.get(t);
			if (orig != null) {
				if (orig.getFlowA() != null) flowA[i] = orig.getFlowA();
				if (orig.getFlowB() != null) flowB[i] = orig.getFlowB();
				if (orig.getLevelA() != null) levelA[i] = orig.getLevelA();
				if (orig.getLevelB() != null) levelB[i] = orig.getLevelB();
			}
		}
		
		System.out.println("[imputate] origin 데이터로 1440개의 배열 생성");
		
		// 2) 결측치 보간
		ImputationConfig impConfig = new ImputationConfig();
		flowA = TmsImputateService.imputeMissingWithStrategy(flowA, impConfig);
		flowB = TmsImputateService.imputeMissingWithStrategy(flowB, impConfig);
		levelA = TmsImputateService.imputeMissingWithStrategy(levelA, impConfig);
		levelB = TmsImputateService.imputeMissingWithStrategy(levelB, impConfig);
		
		System.out.println("[imputate] 데이터 별로 결측치 보간");
		
		// 3) 이상치 탐지 및 처리
		OutlierConfig outConfig = new OutlierConfig();
		flowA = TmsImputateService.detectAndHandleOutliers(flowA, "flowA", outConfig);
		flowB = TmsImputateService.detectAndHandleOutliers(flowB, "flowB", outConfig);
		levelA = TmsImputateService.detectAndHandleOutliers(levelA, "levelA", outConfig);
		levelB = TmsImputateService.detectAndHandleOutliers(levelB, "levelB", outConfig);
		
		System.out.println("[imputate] 이상치 처리");
		
		// 4) List<FlowImputate>으로 변환
		List<FlowImputate> result = new ArrayList<>();
		for (int i = 0; i < 1440; i++) {
			FlowImputate t = new FlowImputate();
			t.setFlowTime(times.get(i));
			t.setFlowA(Double.isNaN(flowA[i]) ? null : flowA[i]);
			t.setFlowB(Double.isNaN(flowB[i]) ? null : flowB[i]);
			t.setLevelA(Double.isNaN(levelA[i]) ? null : levelA[i]);
			t.setLevelB(Double.isNaN(levelB[i]) ? null : levelB[i]);
			result.add(t);
		}
		
		System.out.println("[imputate] 데이터 구성=" + result.size());
		checkNullValues(result);
		return result;
	}
	
	/**
	 * FlowOrigin 리스트의 NULL 값 분석
	 * 각 필드별 NULL 개수와 비율을 출력
	 * 
	 * @param list FlowOrigin 리스트
	 */
	private void checkNullValues(List<FlowImputate> list) {
		if (list == null || list.isEmpty()) {
			System.out.println("[NULL Check] 리스트가 비어있습니다");
			return;
		}
		
		int totalRows = list.size();
		int flowANullCount = 0;
		int flowBNullCount = 0;
		int levelANullCount = 0;
		int levelBNullCount = 0;
		
		// NULL 값 개수 계산
		for (FlowImputate flow : list) {
			if (flow.getFlowA() == null) flowANullCount++;
			if (flow.getFlowB() == null) flowBNullCount++;
			if (flow.getLevelA() == null) levelANullCount++;
			if (flow.getLevelB() == null) levelBNullCount++;
		}
		
		// 결과 출력
		System.out.println("=== FlowOrigin NULL 값 분석 ===");
		System.out.println("총 행 수: " + totalRows);
		System.out.println();
		System.out.printf("flowA   - NULL: %4d / %4d (%.2f%%)%n", flowANullCount, totalRows, (double) flowANullCount / totalRows * 100);
		System.out.printf("flowB    - NULL: %4d / %4d (%.2f%%)%n", flowBNullCount, totalRows, (double) flowBNullCount / totalRows * 100);
		System.out.printf("levelA    - NULL: %4d / %4d (%.2f%%)%n", levelANullCount, totalRows, (double) levelANullCount / totalRows * 100);
		System.out.printf("levelB  - NULL: %4d / %4d (%.2f%%)%n", levelBNullCount, totalRows, (double) levelBNullCount / totalRows * 100);
		System.out.println();
		
		// 전체 NULL 개수
		int totalNulls = flowANullCount + flowBNullCount + levelANullCount + levelBNullCount;
		int totalFields = totalRows * 6;
		System.out.printf("전체 NULL: %d / %d (%.2f%%)%n", totalNulls, totalFields, (double) totalNulls / totalFields * 100);
		System.out.println("==============================");
	}
	
	/**
	 * FlowOrigin 리스트를 CSV 파일로 저장
	 * 
	 * @param list FlowOrigin 리스트
	 * @param filePath 저장할 파일 경로 (상대 경로 또는 절대 경로)
	 * @return 저장 성공 여부
	 * @throws Exception 파일 저장 중 발생하는 예외
	 */
	public boolean saveToCsv(List<FlowImputate> list, String filePath) throws Exception {
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
				bw.write("SYS_TIME,flow_TankA,flow_TankB,level_TankA,level_TankB,Q_in");
				bw.newLine();
				
				// 데이터 작성
				for (FlowImputate flow : list) {
					StringBuilder sb = new StringBuilder();
					//sb.append(flow.getFlowNo()).append(",");
					sb.append(Util.formatDateTime(flow.getFlowTime())).append(",");
					sb.append(Util.formatDouble(flow.getFlowA())).append(",");
					sb.append(Util.formatDouble(flow.getFlowB())).append(",");
					sb.append(Util.formatDouble(flow.getLevelA())).append(",");
					sb.append(Util.formatDouble(flow.getLevelB())).append(",");
					sb.append(Util.formatDouble(flow.getSum()));
					
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
	 * CSV 파일로부터 FlowOrigin 리스트를 로드
	 * 
	 * @param filePath 로드할 파일 경로 (상대 경로 또는 절대 경로)
	 * @return 로드된 FlowOrigin 리스트
	 * @throws Exception 파일 로드 중 발생하는 예외
	 */
	public List<FlowImputate> loadFromCsv(String filePath) {
		List<FlowImputate> list = new ArrayList<>();
		
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
					if (lineNo == 1 && line.contains("flowNo")) {
						continue;
					}
					
					// CSV 파싱
					String[] cols = line.split(",");
					if (cols.length < 8) {
						System.out.println("[loadFromCsv] 경고: 라인 " + lineNo + "의 컬럼 수가 부족합니다. 스킵");
						continue;
					}
					
					try {
						Long flowNo = Util.parseLongOrNull(cols[0]);
						LocalDateTime flowTime = Util.parseDateTime(cols[1]);
						Double flowA = Util.parseDoubleOrNullEmptyOk(cols[2]);
						Double flowB = Util.parseDoubleOrNullEmptyOk(cols[3]);
						Double levelA = Util.parseDoubleOrNullEmptyOk(cols[4]);
						Double levelB = Util.parseDoubleOrNullEmptyOk(cols[5]);
						
						FlowImputate flow = FlowImputate.builder()
							.flowNo(flowNo)
							.flowTime(flowTime)
							.flowA(flowA)
							.flowB(flowB)
							.levelA(levelA)
							.levelB(levelB)
							.build();
						
						list.add(flow);
						
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
	
	public void saveFlowImputateList(List<FlowImputate> list) {
		if(list == null || list.size() == 0) return;
		List<FlowImputate> addList = new ArrayList<>();
		for(FlowImputate tms : list) {
			if(!flowImputateRepo.existsByFlowTime(tms.getFlowTime())) {
				addList.add(tms);
			}
		}
		insertRepo.FlowImputateInsert(addList);
	}
}