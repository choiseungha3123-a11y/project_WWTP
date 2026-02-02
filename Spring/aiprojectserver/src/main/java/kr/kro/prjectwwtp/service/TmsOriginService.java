package kr.kro.prjectwwtp.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import kr.kro.prjectwwtp.domain.TmsLog;
import kr.kro.prjectwwtp.domain.TmsOrigin;
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

}
