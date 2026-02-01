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

import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.persistence.TmsOriginRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class TmsOriginService {
	private final TmsOriginRepository tmsOriginRepo;

	/**
	 * Parse CSV file and save TmsOrigin entries.
	 * Returns detailed import statistics in TmsImportResult.
	 */
	@Transactional
	public TmsImportResult saveFromCsv(MultipartFile file) throws Exception {
		TmsImportResult result = TmsImportResult.builder().totalLines(0).savedCount(0)
				.skippedEmptyOrComment(0).skippedMalformed(0).skippedParseError(0).detectedHeader(false).build();

		if (file == null || file.isEmpty()) return result;

		int lineNo = 0;
		List<TmsOrigin> list = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream(), "UTF-8"))) {
			String firstLine = br.readLine();
			if (firstLine == null) return result;
			lineNo = 1;
			result.setTotalLines(result.getTotalLines() + 1);
			firstLine = firstLine.trim();

			boolean isHeader = false;
			String[] headerCols = null;
			// detect header: contains SYS_TIME or any token with _VU
			String[] firstTokens = splitColumns(firstLine);
			for (String tok : firstTokens) {
				String up = tok.trim().toUpperCase();
				if (up.equals("SYS_TIME") || up.endsWith("_VU") || up.contains("_VU")) {
					isHeader = true;
					result.setDetectedHeader(true);
					break;
				}
			}

			int idxTms = -1, idxToc = -1, idxPh = -1, idxSs = -1, idxFlux = -1, idxTn = -1, idxTp = -1;
			if (isHeader) {
				headerCols = firstTokens;
				for (int i = 0; i < headerCols.length; ++i) {
					String h = headerCols[i].trim().toUpperCase();
					// normalize quotes
					h = h.replaceAll("\"", "").replaceAll("'", "");
					if (h.equals("SYS_TIME")) idxTms = i;
					if (h.endsWith("_VU")) {
						String base = h.substring(0, h.length() - 3); // remove _VU
						switch (base) {
							case "TOC": idxToc = i; break;
							case "PH": idxPh = i; break;
							case "SS": idxSs = i; break;
							case "FLUX": idxFlux = i; break;
							case "TN": idxTn = i; break;
							case "TP": idxTp = i; break;
							default:
								if (base.contains("TOC")) idxToc = i;
								else if (base.contains("PH")) idxPh = i;
								else if (base.contains("SS")) idxSs = i;
								else if (base.contains("FLUX")) idxFlux = i;
								else if (base.contains("TN")) idxTn = i;
								else if (base.contains("TP")) idxTp = i;
								break;
						}
					}
				}
			} else {
				// first line is data, process it below like normal lines
				boolean ok = processDataLineCollect(firstLine, ++lineNo, list);
				result.setTotalLines(result.getTotalLines() + 1);
				if (!ok) result.setSkippedMalformed(result.getSkippedMalformed() + 1);
			}

			String line;
			while ((line = br.readLine()) != null) {
				++lineNo;
				result.setTotalLines(result.getTotalLines() + 1);
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) {
					result.setSkippedEmptyOrComment(result.getSkippedEmptyOrComment() + 1);
					continue;
				}
				// parse using header mapping if present
				if (isHeader) {
					String[] cols = splitColumns(line);
					// map by indices - if index out of bounds, treat as null
					try {
						String timeStr = (idxTms >= 0 && idxTms < cols.length) ? cols[idxTms].trim() : null;
						LocalDateTime tmsTime = (timeStr == null || timeStr.length() == 0) ? null : parseDateTime(timeStr);
						Double toc = (idxToc >= 0 && idxToc < cols.length) ? parseDoubleOrNullEmptyOk(cols[idxToc]) : null;
						Double ph = (idxPh >= 0 && idxPh < cols.length) ? parseDoubleOrNullEmptyOk(cols[idxPh]) : null;
						Double ss = (idxSs >= 0 && idxSs < cols.length) ? parseDoubleOrNullEmptyOk(cols[idxSs]) : null;
						Integer flux = (idxFlux >= 0 && idxFlux < cols.length) ? parseIntOrNullEmptyOk(cols[idxFlux]) : null;
						Double tn = (idxTn >= 0 && idxTn < cols.length) ? parseDoubleOrNullEmptyOk(cols[idxTn]) : null;
						Double tp = (idxTp >= 0 && idxTp < cols.length) ? parseDoubleOrNullEmptyOk(cols[idxTp]) : null;
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
					} catch (Exception e) {
						// skip line on parse error
						result.setSkippedParseError(result.getSkippedParseError() + 1);
						continue;
					}
				} else {
					// no header - positional parsing
					boolean ok = processDataLineCollect(line, lineNo, list);
					if (!ok) result.setSkippedMalformed(result.getSkippedMalformed() + 1);
				}
			}
		}

		if (!list.isEmpty()) {
			tmsOriginRepo.saveAll(list);
			result.setSavedCount(list.size());
		}
		return result;
	}

	// helper: split by comma, trimming; fallback keeps original order
	private String[] splitColumns(String line) {
		String[] cols = line.split(",");
		if (cols.length <= 1) cols = line.split("\\s+");
		for (int i = 0; i < cols.length; ++i) cols[i] = cols[i].trim();
		return cols;
	}

	// helper to process a positional data line (time, toc, ph, ss, flux, tn, tp)
	// returns true if parsed and added to list, false otherwise
	private boolean processDataLineCollect(String line, int lineNo, List<TmsOrigin> list) {
		String[] cols = splitColumns(line);
		if (cols.length < 7) return false; // malformed
		try {
			String timeStr = cols[0];
			LocalDateTime tmsTime = (timeStr == null || timeStr.trim().length() == 0) ? null : parseDateTime(timeStr);
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
			return true;
		} catch (Exception e) {
			return false;
		}
	}

	// parse helpers that treat empty string as null (explicit)
	private Double parseDoubleOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null") || t.equalsIgnoreCase("-99.0") || t.equalsIgnoreCase("-99.9")) return null;
		return Double.parseDouble(t);
	}

	private Integer parseIntOrNullEmptyOk(String s) {
		if (s == null) return null;
		String t = s.trim();
		if (t.length() == 0) return null;
		if (t.equalsIgnoreCase("NA") || t.equalsIgnoreCase("null")) return null;
		return Integer.parseInt(t);
	}

	private LocalDateTime parseDateTime(String s) {
		String str = s.trim();
		// try several common patterns
		String[] patterns = new String[] {
			"yyyyMMddHHmm",
			"yyyyMMddHHmmss",
			"yyyy-MM-dd HH:mm:ss",
			"yyyy/MM/dd HH:mm:ss",
			"yyyy-MM-dd'T'HH:mm:ss",
			"yyyy-MM-dd'T'HH:mm:ss.SSS",
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
