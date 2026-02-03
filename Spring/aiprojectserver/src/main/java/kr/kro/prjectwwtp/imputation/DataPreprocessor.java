package kr.kro.prjectwwtp.imputation;

public class DataPreprocessor {
/*	
	public Table imputeMissingWithStrategy(Table df, String freq, ImputationConfig config, boolean addMask) {
        Table dfOut = df.copy();
        double freqHours = Duration.parse("PT" + freq.toUpperCase()).toMinutes() / 60.0; //[cite: 4]

        for (String colName : df.columnNames()) {
            DoubleColumn series = df.nCol(colName).asDoubleColumn();
            if (series.countMissing() == 0) continue; //[cite: 5]

            // 1단계: Forward Fill (단기 결측) [cite: 6]
            int limitShort = (int) (config.getShortTermHours() / freqHours);
            DoubleColumn ffillSeries = applyForwardFill(series, limitShort);
            
            // 2단계: EWMA (중기 결측) [cite: 7, 9]
            int limitMedium = (int) (config.getMediumTermHours() / freqHours);
            DoubleColumn ewmaSeries = applyEWMA(ffillSeries, config.getEwmaSpan());
            // (중기 마스크 로직 및 적용 부분 생략 - Tablesaw 반복문으로 구현 가능) [cite: 9]

            // 3단계: Rolling Median (장기 결측) [cite: 10, 11]
            int rollingWindow = (int) (config.getRollingWindow() / freqHours);
            // Tablesaw의 rolling() 기능을 사용하여 중앙값 계산 [cite: 10]
            
            dfOut.replaceColumn(colName, ffillSeries);
        }
        return dfOut; //[cite: 12]
    }
    
    // Forward Fill 구현 (제한적 적용) [cite: 6]
    private DoubleColumn applyForwardFill(DoubleColumn col, int limit) {
        DoubleColumn result = col.copy();
        int count = 0;
        for (int i = 1; i < result.size(); i++) {
            if (Double.isNaN(result.get(i))) {
                if (count < limit && !Double.isNaN(result.get(i - 1))) {
                    result.set(i, result.get(i - 1));
                    count++;
                }
            } else {
                count = 0;
            }
        }
        return result;
    }
    
    // 지수 이동 평균(EWMA)을 계산하여 결측치를 보간합니다.
    // 파이썬의 series.ewm(span=span).mean()과 유사하게 동작합니다.
    public static DoubleColumn applyEWMA(DoubleColumn col, int span) {
        DoubleColumn result = col.copy();
        double alpha = 2.0 / (span + 1.0);
        double lastEma = Double.NaN;

        for (int i = 0; i < result.size(); i++) {
            double currentVal = result.get(i);

            if (!Double.isNaN(currentVal)) {
                // 현재 값이 있으면 EMA 업데이트
                if (Double.isNaN(lastEma)) {
                    lastEma = currentVal; // 첫 번째 유효한 값으로 초기화
                } else {
                    lastEma = alpha * currentVal + (1 - alpha) * lastEma;
                }
            } else {
                // 현재 값이 NaN(결측치)이면 계산된 lastEma로 보간
                if (!Double.isNaN(lastEma)) {
                    result.set(i, lastEma);
                    // 보간된 값으로 다음 EMA를 업데이트 (과거 추세 유지)
                    lastEma = alpha * lastEma + (1 - alpha) * lastEma; 
                }
            }
        }
        return result;
    }
*/    
}
