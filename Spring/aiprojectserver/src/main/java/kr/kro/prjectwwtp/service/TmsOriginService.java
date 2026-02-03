package kr.kro.prjectwwtp.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import kr.kro.prjectwwtp.domain.TmsLog;
import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.imputation.ImputationConfig;
import kr.kro.prjectwwtp.imputation.OutlierConfig;
import kr.kro.prjectwwtp.imputation.OutlierHandler;
import kr.kro.prjectwwtp.imputation.TmsDataProcessor;
import kr.kro.prjectwwtp.persistence.TmsLogRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginInsertRepository;
import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import lombok.RequiredArgsConstructor;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

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
		// 1. 샘플 데이터 준비
		LocalDateTime start = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay();
		LocalDateTime end = LocalDateTime.of(start.getYear(), start.getMonth(), start.getDayOfMonth(), 23, 59, 59);
		List<TmsOrigin> origin = tmsOriginRepo.findByTmsTimeBetween(start, end);
		
		// 2. 설정 객체 생성
		ImputationConfig impConfig = ImputationConfig.builder()
                .shortTermHours(3)    // 3시간 이내 FFill [cite: 1, 6]
                .mediumTermHours(12)  // 12시간 이내 EWMA [cite: 1, 8]
                .ewmaSpan(6)
                .rollingWindow(24)
                .build();
		
		OutlierConfig outConfig = OutlierConfig.builder()
                .method("iqr")        // IQR 통계 방식 사용 [cite: 12, 23]
                .iqrThreshold(1.5)
                .requireBoth(true)    // 도메인과 통계 모두 이상치일 때만 처리 (보수적) [cite: 12, 27]
                .ewmaSpan(12)
                .build();
		
		// 3. 전처리 인스턴스 생성
		TmsDataProcessor preprocessor = new TmsDataProcessor();
        OutlierHandler handler = new OutlierHandler();
        
        try {
        	// STEP 1: 1분 단위 테이블 생성 및 결측치 보간
            // 전략적 보간 실행 (Forward Fill -> EWMA -> Rolling Median)
            Table imputedTable = preprocessor.processTmsData(origin, start, impConfig);
            System.out.println("결측치 보간 완료: " + imputedTable.rowCount() + "행 생성");
            
            // STEP 2: 이상치 탐지 및 EWMA 대체
            // 도메인 규칙과 통계적 방법을 병행하여 이상치 처리
            Table finalTable = handler.detectAndHandleOutliers(imputedTable, outConfig, true);
            System.out.println("이상치 처리 완료");
            
			// 5. 최종 데이터 활용 (CSV 저장 등)
            
            List<TmsOrigin> list = convertTableToList(finalTable);
			return list;
            
        } catch (Exception e) {
            e.printStackTrace();
        }
		return null;
	}
	
	static public List<TmsOrigin> convertTableToList(Table table) {
	    List<TmsOrigin> resultList = new ArrayList<>();

	    // 테이블의 각 행을 순회
	    for (Row row : table) {
	        TmsOrigin tms = new TmsOrigin();
	        
	        // 1. 시간 매핑 (DateTimeColumn -> LocalDateTime)
	        tms.setTmsTime(row.getDateTime("tmsTime"));

	        // 2. 수치 데이터 매핑 (DoubleColumn -> Double)
	        // Tablesaw의 row.getDouble()은 값이 NaN일 때 매우 작은 수나 특이값을 반환할 수 있으므로 
	        // 직접 컬럼에서 인덱스로 접근하거나 NaN 체크를 해주는 것이 좋습니다.
	        if(getNullableDouble(row, "toc") == null)
	        	continue;
	        tms.setToc(getNullableDouble(row, "toc"));
	        tms.setPh(getNullableDouble(row, "ph"));
	        tms.setSs(getNullableDouble(row, "ss"));
	        tms.setFlux(getNullableInt(row, "flux"));
	        tms.setTn(getNullableDouble(row, "tn"));
	        tms.setTp(getNullableDouble(row, "tp"));

	        resultList.add(tms);
	    }
	    
	    return resultList;
	}
	
	static public Double getNullableDouble(Row row, String columnName) {
	    if (row.isMissing(columnName)) {
	        return null;
	    }
	    double value = row.getDouble(columnName);
	    return Double.isNaN(value) ? null : value;
	}
	
	static public Integer getNullableInt(Row row, String columnName) {
	    if (row.isMissing(columnName)) {
	        return null;
	    }
	    int value = (int) row.getDouble(columnName);
	    return Double.isNaN(value) ? null : value;
	}
    
    // Tablesaw 컬럼의 특정 인덱스가 결측치(NaN)인지 또는 실제 null인지 통합 체크
	static public boolean isMissing(DoubleColumn col, int index) {
        // Tablesaw의 DoubleColumn은 내부적으로 null을 NaN으로 처리하지만,
        // 데이터 소스에 따라 발생할 수 있는 모든 null 케이스를 방어합니다.
        return col.isMissing(index) || Double.isNaN(col.get(index));
    }
	
	// 지수 이동 평균(EWMA)을 계산하여 결측치를 보간합니다.
    // 파이썬의 series.ewm(span=span).mean()과 유사하게 동작합니다.
	static public DoubleColumn applyEWMA(DoubleColumn col, int span) {
	    DoubleColumn result = col.copy();
	    double alpha = 2.0 / (span + 1.0);
	    Double lastEma = null; // 초기 EMA 값을 객체형으로 관리하여 null 체크 가능하게 함

	    for (int i = 0; i < result.size(); i++) {
	        // 1. 현재 행의 값이 결측치(NaN/Null)인지 체크
	        if (!isMissing(result, i)) {
	            // [유효한 데이터인 경우]
	            double currentVal = result.get(i);
	            
	            if (lastEma == null) {
	                // 첫 번째 유효한 값으로 EMA 초기화
	                lastEma = currentVal;
	            } else {
	                // EMA 업데이트 수식 적용: EMA_t = α * X_t + (1 - α) * EMA_{t-1}
	                lastEma = alpha * currentVal + (1 - alpha) * lastEma;
	            }
	        } else {
	            // [결측치(NaN/Null)인 경우]
	            // 계산된 이전 EMA 값이 존재한다면 그 값으로 결측치를 채움
	            if (lastEma != null) {
	                result.set(i, lastEma);
	                
	                // 보간된 값을 바탕으로 다음 단계 EMA 추세 유지 (옵션)
	                // 파이썬의 ewm().mean()과 유사하게 동작하도록 추세 가중치를 유지함
	                lastEma = alpha * lastEma + (1 - alpha) * lastEma; 
	            }
	        }
	    }
	    return result;
	}

}
