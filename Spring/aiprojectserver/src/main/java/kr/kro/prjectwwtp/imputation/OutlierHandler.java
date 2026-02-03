package kr.kro.prjectwwtp.imputation;

import java.util.Arrays;

import kr.kro.prjectwwtp.service.TmsOriginService;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;

public class OutlierHandler {
    
    // 도메인 지식 기반 탐지 [cite: 14]
    public boolean[] detectOutliersDomain(DoubleColumn series, String colName) {
        boolean[] outliers = new boolean[series.size()];
        
        // 도메인 규칙 정의 [cite: 15, 16]
        if (colName.equals("PH_VU")) {
            for (int i = 0; i < series.size(); i++) {
                double val = series.get(i);
                outliers[i] = (val < 0 || val > 14); //[cite: 15, 18]
            }
        } else if (colName.contains("RN_")) { // 강수량 [cite: 19]
            for (int i = 0; i < series.size(); i++) {
                double val = series.get(i);
                outliers[i] = (val < 0 || val > 300); //[cite: 19]
            }
        }
        return outliers;
    }

    // 통계적 탐지 (IQR) [cite: 22, 24]
    public boolean[] detectOutliersStatistical(DoubleColumn series, OutlierConfig config) {
        int n = series.size();
        boolean[] outliers = new boolean[n];
        
        // 1. 유효 데이터만 추출 (NaN/Missing 완전 제거)
        // Tablesaw의 removeMissing()은 내부적으로 isMissing을 모두 걸러냅니다.
        DoubleColumn validValues = series.removeMissing();
        if (validValues.isEmpty()) return outliers;

        if ("iqr".equalsIgnoreCase(config.getMethod())) {
            // percentile(25), percentile(75) 사용
            double q1 = validValues.percentile(25);
            double q3 = validValues.percentile(75);
            double iqr = q3 - q1;
            
            double lower = q1 - config.getIqrThreshold() * iqr;
            double upper = q3 + config.getIqrThreshold() * iqr;
            
            for (int i = 0; i < n; i++) {
                // 현재 값이 결측치(null/NaN)인 경우는 이상치 판정에서 제외
                if (!TmsOriginService.isMissing(series, i)) {
                    double val = series.get(i);
                    outliers[i] = (val < lower || val > upper);
                }
            }
        } else if ("zscore".equalsIgnoreCase(config.getMethod())) {
            double mean = validValues.mean();
            double std = validValues.standardDeviation();
            
            if (std > 0) {
                for (int i = 0; i < n; i++) {
                    if (!TmsOriginService.isMissing(series, i)) {
                        double zScore = Math.abs((series.get(i) - mean) / std);
                        outliers[i] = zScore > config.getZscoreThreshold();
                    }
                }
            }
        }
        return outliers;
    }

    //이상치 탐지 및 처리 (EWMA로 대체)
    //전략: 도메인 지식 + 통계적 방법 병행 
    public Table detectAndHandleOutliers(Table df, OutlierConfig config, boolean addMask) {
        Table dfOut = df.copy();
        
        // 처리에서 제외할 컬럼 패턴 정의 [cite: 28, 29]
        String[] skipPatterns = {"_is_missing", "_imputed_", "_outlier_"};

        for (String colName : df.columnNames()) {
            // 마스크 컬럼 및 숫자형이 아닌 컬럼 건너뛰기 [cite: 28, 29]
            if (Arrays.stream(skipPatterns).anyMatch(colName::contains)) continue;
            Column<?> column = df.column(colName);
            if (!(column instanceof DoubleColumn)) continue;

            DoubleColumn series = ((DoubleColumn) column).copy();
            if (series.countMissing() == series.size()) continue; // 모두 결측인 경우 스킵 [cite: 29]

            // 1. 도메인 지식 기반 이상치 탐지 [cite: 30]
            boolean[] domainOutliers = detectOutliersDomain(series, colName);
            
            // 2. 통계적 방법 기반 이상치 탐지 [cite: 30]
            boolean[] statisticalOutliers = detectOutliersStatistical(series, config);
            
            // 3. 최종 이상치 결정 (requireBoth 설정에 따라 결정) [cite: 31, 32]
            boolean[] finalOutliers = new boolean[series.size()];
            int outlierCount = 0;
            for (int i = 0; i < series.size(); i++) {
                if (config.isRequireBoth()) {
                    finalOutliers[i] = domainOutliers[i] && statisticalOutliers[i]; // 둘 다 해당 시 [cite: 31]
                } else {
                    finalOutliers[i] = domainOutliers[i] || statisticalOutliers[i]; // 하나만 해당 시 
                }
                if (finalOutliers[i]) outlierCount++;
            }

            // 4. 마스크 컬럼 추가 (디버깅용) 
            if (addMask) {
                dfOut.addColumns(IntColumn.create(colName + "_outlier_final", 
                    convertBooleanToInt(finalOutliers)));
            }

            // 5. 이상치를 EWMA로 대체 
            if (outlierCount > 0) {
                // 이상치 위치를 NaN으로 변환 [cite: 33]
                for (int i = 0; i < series.size(); i++) {
                    if (finalOutliers[i]) {
                        series.set(i, Double.NaN);
                    }
                }

                // EWMA 계산 (과거 데이터 기반 추세값) [cite: 33]
                // 앞서 구현한 applyEWMA 메서드 활용
                DoubleColumn ewmaSeries = TmsOriginService.applyEWMA(series, config.getEwmaSpan());

                // 이상치 위치만 EWMA 값으로 대체 [cite: 34]
                for (int i = 0; i < series.size(); i++) {
                    if (finalOutliers[i]) {
                        series.set(i, ewmaSeries.get(i));
                    }
                }
                dfOut.replaceColumn(colName, series);
            }
        }
        return dfOut;
    }

    private int[] convertBooleanToInt(boolean[] boolArray) {
        int[] intArray = new int[boolArray.length];
        for (int i = 0; i < boolArray.length; i++) {
            intArray[i] = boolArray[i] ? 1 : 0;
        }
        return intArray;
    }
}
