package kr.kro.prjectwwtp.service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 결측치 및 이상치 처리 통합 클래스
 * 
 * 파이썬 코드를 Java로 변환:
 * - ImputationConfig: 결측치 보간 설정
 * - OutlierConfig: 이상치 탐지 설정
 * - ImputationStrategy: 결측치 보간 전략 (forwardFill, backwardFill, EWMA, rolling median)
 * - OutlierDetector: 이상치 탐지 (도메인 기반, 통계 기반)
 * - OutlierHandler: 이상치 처리 (EWMA로 대체)
 */
public class TmsImputateService {

	// ==================== 설정 클래스 ====================

	/**
	 * 결측치 보간 설정
	 */
	public static class ImputationConfig {
		/** 단기 결측 범위 (시간) */
		public int shortTermHours = 3;
		
		/** 중기 결측 범위 (시간) */
		public int mediumTermHours = 12;
		
		/** EWMA 스팬 (시간 단위) */
		public int ewmaSpan = 6;
		
		/** Rolling median 윈도우 (시간 단위, 장기 결측용) */
		public int rollingWindow = 24;

		public ImputationConfig() {}

		public ImputationConfig(int shortTermHours, int mediumTermHours, int ewmaSpan, int rollingWindow) {
			this.shortTermHours = shortTermHours;
			this.mediumTermHours = mediumTermHours;
			this.ewmaSpan = ewmaSpan;
			this.rollingWindow = rollingWindow;
		}
	}

	/**
	 * 이상치 탐지 설정
	 */
	public static class OutlierConfig {
		/** 이상치 탐지 방법: 'iqr' 또는 'zscore' */
		public String method = "iqr";
		
		/** IQR 배수 (기본: 1.5) */
		public double iqrThreshold = 1.5;
		
		/** Z-score 임계값 (기본: 3.0) */
		public double zscoreThreshold = 3.0;
		
		/** 도메인+통계 둘 다 이상치여야 처리할지 여부 */
		public boolean requireBoth = true;
		
		/** 이상치 대체용 EWMA 스팬 */
		public int ewmaSpan = 12;

		public OutlierConfig() {}

		public OutlierConfig(String method, double iqrThreshold, double zscoreThreshold, boolean requireBoth, int ewmaSpan) {
			this.method = method;
			this.iqrThreshold = iqrThreshold;
			this.zscoreThreshold = zscoreThreshold;
			this.requireBoth = requireBoth;
			this.ewmaSpan = ewmaSpan;
		}
	}

	// ==================== 결측치 보간 전략 ====================

	/**
	 * 전략적 결측치 보간
	 * 
	 * 전략:
	 * - 단기 결측 (1-3시간): Forward Fill
	 * - 중기 결측 (4-12시간): EWMA (시간 가중 이동평균, 과거 데이터만 사용)
	 * - 장기 결측 (12시간+): Rolling Median (중앙값 기반)
	 * 
	 * @param data 결측치가 있는 배열 (NaN으로 표현)
	 * @param config 보간 설정
	 * @return 보간된 배열
	 */
	public static double[] imputeMissingWithStrategy(double[] data, ImputationConfig config) {
		double[] result = Arrays.copyOf(data, data.length);
		int shortMinutes = config.shortTermHours * 3600;
		int mediumMinutes = config.mediumTermHours * 3600;
		int rollingMinutes = config.rollingWindow * 3600;
		
		// 1단계: Forward Fill (단기 결측, limit=short_term_hours)
		forwardFill(result, shortMinutes);
		
		// 2단계: EWMA (중기 결측, 단기 < 길이 <= 중기)
		boolean[] stillMissing = new boolean[result.length];
		for (int i = 0; i < result.length; i++) {
			stillMissing[i] = Double.isNaN(result[i]);
		}
		
		if (anyTrue(stillMissing)) {
			double[] ewmaValues = applyEWMA(result, config.ewmaSpan);
			int[] groupLengths = getConsecutiveMissingLengths(result);
			
			for (int i = 0; i < result.length; i++) {
				if (Double.isNaN(result[i]) && groupLengths[i] > shortMinutes && groupLengths[i] <= mediumMinutes && !Double.isNaN(ewmaValues[i])) {
					result[i] = ewmaValues[i];
				}
			}
		}
		
		// 3단계: Rolling Median (장기 결측, center=True로 앞뒤 데이터 모두 사용)
		// min_periods=1: 1개 값만 있어도 중앙값 계산
		boolean[] stillMissingLong = new boolean[result.length];
		for (int i = 0; i < result.length; i++) {
			stillMissingLong[i] = Double.isNaN(result[i]);
		}
		
		if (anyTrue(stillMissingLong)) {
			int half = Math.max(1, rollingMinutes / 2);
			for (int i = 0; i < result.length; i++) {
				if (Double.isNaN(result[i])) {
					int start = Math.max(0, i - half);
					int end = Math.min(result.length - 1, i + half);
					// min_periods=1: 최소 1개 값이 있으면 중앙값 계산
					double median = getMedianMinPeriods(result, start, end, 1);
					if (!Double.isNaN(median)) {
						result[i] = median;
					}
				}
			}
		}
		
		// 4단계: 최후의 수단 - 선형 보간 (극도의 결측만 처리)
		// 모든 앞 단계에서도 채워지지지 않은 NaN 값에 대해 선형 보간
		boolean[] stillMissingFinal = new boolean[result.length];
		for (int i = 0; i < result.length; i++) {
			stillMissingFinal[i] = Double.isNaN(result[i]);
		}
		
		if (anyTrue(stillMissingFinal)) {
			linearInterpolation(result);
		}
		
		return result;
	}

	/**
	 * Forward Fill: 과거 값으로 채우기
	 * 
	 * @param data 배열
	 * @param limitMinutes 채우기 제한 분 단위
	 */
	public static void forwardFill(double[] data, int limitMinutes) {
		double lastVal = Double.NaN;
		int lastIdx = -100000;
		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) {
				lastVal = data[i];
				lastIdx = i;
			} else if (!Double.isNaN(lastVal) && (i - lastIdx) <= limitMinutes) {
				data[i] = lastVal;
			}
		}
	}

	/**
	 * EWMA (Exponentially Weighted Moving Average) 적용
	 * 시간 가중 이동평균으로 중기 결측 보간
	 * 
	 * @param data 배열
	 * @param span 스팬 (분 단위)
	 * @return EWMA 값들
	 */
	public static double[] applyEWMA(double[] data, int span) {
		double[] result = Arrays.copyOf(data, data.length);
		double alpha = 2.0 / (span + 1.0);
		Double lastEma = null;
		
		for (int i = 0; i < result.length; i++) {
			if (!Double.isNaN(result[i])) {
				double current = result[i];
				lastEma = (lastEma == null) ? current : (alpha * current + (1 - alpha) * lastEma);
			} else if (lastEma != null) {
				result[i] = lastEma;
			}
		}
		
		return result;
	}

	/**
	 * 연속된 결측 구간의 길이 계산
	 * 
	 * @param data 배열
	 * @return 각 위치의 결측 구간 길이
	 */
	public static int[] getConsecutiveMissingLengths(double[] data) {
		int[] lengths = new int[data.length];
		for (int i = 0; i < data.length; i++) {
			if (Double.isNaN(data[i])) {
				int j = i;
				while (j < data.length && Double.isNaN(data[j])) {
					j++;
				}
				int len = j - i;
				for (int k = i; k < j; k++) {
					lengths[k] = len;
				}
				i = j - 1;
			}
		}
		return lengths;
	}

	/**
	 * 중앙값(Median) 계산
	 * 
	 * @param data 배열
	 * @param start 시작 인덱스
	 * @param end 종료 인덱스
	 * @return 중앙값
	 */
	public static double getMedian(double[] data, int start, int end) {
		List<Double> values = new ArrayList<>();
		for (int i = start; i <= end; i++) {
			if (!Double.isNaN(data[i])) {
				values.add(data[i]);
			}
		}
		if (values.isEmpty()) return Double.NaN;
		Collections.sort(values);
		int mid = values.size() / 2;
		if (values.size() % 2 == 1) {
			return values.get(mid);
		} else {
			return (values.get(mid - 1) + values.get(mid)) / 2.0;
		}
	}

	/**
	 * 중앙값(Median) 계산 with min_periods
	 * 파이썬의 rolling(min_periods=1)과 동일
	 * 최소 min_periods개의 유효한 값이 있으면 중앙값 계산
	 * 
	 * @param data 배열
	 * @param start 시작 인덱스
	 * @param end 종료 인덱스
	 * @param minPeriods 최소 필요 값의 개수
	 * @return 중앙값 (min_periods 미만이면 NaN)
	 */
	public static double getMedianMinPeriods(double[] data, int start, int end, int minPeriods) {
		List<Double> values = new ArrayList<>();
		for (int i = start; i <= end; i++) {
			if (!Double.isNaN(data[i])) {
				values.add(data[i]);
			}
		}
		// min_periods 개 미만이면 NaN 반환
		if (values.size() < minPeriods) return Double.NaN;
		if (values.isEmpty()) return Double.NaN;
		
		Collections.sort(values);
		int mid = values.size() / 2;
		if (values.size() % 2 == 1) {
			return values.get(mid);
		} else {
			return (values.get(mid - 1) + values.get(mid)) / 2.0;
		}
	}

	/**
	 * 백분위수(Quantile) 계산
	 * 
	 * @param sorted 정렬된 배열
	 * @param p 백분위수 (0~1)
	 * @return 백분위수 값
	 */
	public static double quantile(double[] sorted, double p) {
		if (sorted.length == 0) return Double.NaN;
		double idx = p * (sorted.length - 1);
		int lo = (int) Math.floor(idx);
		int hi = (int) Math.ceil(idx);
		if (lo == hi) return sorted[lo];
		double w = idx - lo;
		return sorted[lo] * (1 - w) + sorted[hi] * w;
	}

	// ==================== 이상치 탐지 및 처리 ====================

	/**
	 * 이상치 탐지 및 처리 (EWMA로 대체)
	 * 
	 * 전략:
	 * - 도메인 지식 + 통계적 방법 병행
	 * - require_both=true: 둘 다 이상치여야 처리 (보수적)
	 * - require_both=false: 둘 중 하나만 이상치여도 처리 (공격적)
	 * - 이상치는 EWMA로 대체
	 * 
	 * @param data 배열
	 * @param colName 컬럼명
	 * @param config 이상치 탐지 설정
	 * @return 이상치가 EWMA로 대체된 배열
	 */
	public static double[] detectAndHandleOutliers(double[] data, String colName, OutlierConfig config) {
		double[] result = Arrays.copyOf(data, data.length);
		
		// 도메인 이상치 탐지
		boolean[] domainOutliers = detectOutliersDomain(data, colName);
		
		// 통계 이상치 탐지
		boolean[] statOutliers = detectOutliersStatistical(data, config.method, config.iqrThreshold, config.zscoreThreshold);
		
		// 최종 이상치 결정
		boolean[] finalOutliers = new boolean[data.length];
		boolean anyOutlier = false;
		for (int i = 0; i < data.length; i++) {
			finalOutliers[i] = config.requireBoth ? (domainOutliers[i] & statOutliers[i]) : (domainOutliers[i] | statOutliers[i]);
			if (finalOutliers[i]) anyOutlier = true;
		}
		
		if (!anyOutlier) return result;
		
		// 이상치를 NaN으로 변환
		double[] clean = Arrays.copyOf(result, result.length);
		for (int i = 0; i < clean.length; i++) {
			if (finalOutliers[i]) {
				clean[i] = Double.NaN;
			}
		}
		
		// EWMA로 대체
		double[] ewmaValues = applyEWMA(clean, config.ewmaSpan);
		for (int i = 0; i < result.length; i++) {
			if (finalOutliers[i] && !Double.isNaN(ewmaValues[i])) {
				result[i] = ewmaValues[i];
			}
		}
		
		return result;
	}

	/**
	 * 도메인 지식 기반 이상치 탐지
	 * 
	 * @param data 배열
	 * @param colName 컬럼명
	 * @return 이상치 마스크 (true = 이상치)
	 */
	public static boolean[] detectOutliersDomain(double[] data, String colName) {
		boolean[] outliers = new boolean[data.length];
		double lower = Double.NEGATIVE_INFINITY;
		double upper = Double.POSITIVE_INFINITY;
		boolean useRule = false;
		
		String name = colName.toLowerCase();
		if (name.contains("toc")) { 
			lower = 0; upper = 250; useRule = true; 
		} else if (name.contains("ph")) { 
			lower = 0; upper = 14; useRule = true; 
		} else if (name.contains("ss")) { 
			lower = 0; upper = 100; useRule = true; 
		} else if (name.contains("tn")) { 
			lower = 0; upper = 100; useRule = true; 
		} else if (name.contains("tp")) { 
			lower = 0; upper = 20; useRule = true; 
		} else if (name.contains("ta")) { 
			lower = -30; upper = 45; useRule = true; 
		} else if (name.contains("hm")) { 
			lower = 0; upper = 100; useRule = true; 
		} else if (name.contains("td")) { 
			lower = -40; upper = 35; useRule = true; 
		} else if (name.contains("rn") || name.contains("rain")) { 
			// 강수량: 음수 또는 300mm 초과
			lower = 0; upper = 300; useRule = true; 
		} else if (name.contains("flux") || name.contains("flow") || name.contains("q_in")) { 
			// 유량: 음수는 항상 이상치
			lower = 0; useRule = true; 
		} else { 
			lower = 0; upper = 1e6; useRule = true; 
		}
		
		for (int i = 0; i < data.length; i++) {
			if (Double.isNaN(data[i])) {
				outliers[i] = false;
			} else {
				outliers[i] = useRule && ((data[i] < lower) || (data[i] > upper));
			}
		}
		
		return outliers;
	}

	/**
	 * 통계 기반 이상치 탐지
	 * 
	 * @param data 배열
	 * @param method 'iqr' 또는 'zscore'
	 * @param iqrThreshold IQR 배수
	 * @param zscoreThreshold Z-score 임계값
	 * @return 이상치 마스크 (true = 이상치)
	 */
	public static boolean[] detectOutliersStatistical(double[] data, String method, double iqrThreshold, double zscoreThreshold) {
		boolean[] outliers = new boolean[data.length];
		
		// 유효한 값들 수집
		List<Double> validValues = new ArrayList<>();
		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) {
				validValues.add(data[i]);
			}
		}
		
		if (validValues.isEmpty()) return outliers;
		
		double[] sorted = new double[validValues.size()];
		for (int i = 0; i < validValues.size(); i++) {
			sorted[i] = validValues.get(i);
		}
		Arrays.sort(sorted);
		
		if ("iqr".equalsIgnoreCase(method)) {
			double q1 = quantile(sorted, 0.25);
			double q3 = quantile(sorted, 0.75);
			double iqr = q3 - q1;
			double lowerBound = q1 - iqrThreshold * iqr;
			double upperBound = q3 + iqrThreshold * iqr;
			
			for (int i = 0; i < data.length; i++) {
				if (Double.isNaN(data[i])) {
					outliers[i] = false;
				} else {
					outliers[i] = (data[i] < lowerBound) || (data[i] > upperBound);
				}
			}
		} else if ("zscore".equalsIgnoreCase(method)) {
			double mean = 0;
			for (double v : sorted) mean += v;
			mean /= sorted.length;
			
			double variance = 0;
			for (double v : sorted) variance += (v - mean) * (v - mean);
			double sd = Math.sqrt(variance / sorted.length);
			
			for (int i = 0; i < data.length; i++) {
				if (Double.isNaN(data[i])) {
					outliers[i] = false;
				} else {
					double z = sd == 0 ? 0 : Math.abs((data[i] - mean) / sd);
					outliers[i] = z > zscoreThreshold;
				}
			}
		}
		
		return outliers;
	}

	/**
	 * 선형 보간 (Linear Interpolation)
	 * 극도의 결측 (모든 방법에도 채워지지지 않은 NaN)을 처리
	 * 
	 * @param data 배열
	 */
	public static void linearInterpolation(double[] data) {
		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) continue;
			
			// 이전 유효한 값 찾기
			int prevIdx = -1;
			for (int j = i - 1; j >= 0; j--) {
				if (!Double.isNaN(data[j])) {
					prevIdx = j;
					break;
				}
			}
			
			// 다음 유효한 값 찾기
			int nextIdx = -1;
			for (int j = i + 1; j < data.length; j++) {
				if (!Double.isNaN(data[j])) {
					nextIdx = j;
					break;
				}
			}
			
			// 선형 보간
			if (prevIdx != -1 && nextIdx != -1) {
				// 이전과 다음 값 사이에서 선형 보간
				double prevVal = data[prevIdx];
				double nextVal = data[nextIdx];
				double ratio = (double) (i - prevIdx) / (nextIdx - prevIdx);
				data[i] = prevVal + ratio * (nextVal - prevVal);
			} else if (prevIdx != -1) {
				// 이전 값만 있는 경우
				data[i] = data[prevIdx];
			} else if (nextIdx != -1) {
				// 다음 값만 있는 경우
				data[i] = data[nextIdx];
			}
			// else: 값이 없으면 NaN 유지 (극도의 경우)
		}
	}

	/**
	 * 배열에 true 값이 있는지 확인
	 * 
	 * @param arr 배열
	 * @return true 값이 있으면 true
	 */
	private static boolean anyTrue(boolean[] arr) {
		for (boolean b : arr) {
			if (b) return true;
		}
		return false;
	}
}