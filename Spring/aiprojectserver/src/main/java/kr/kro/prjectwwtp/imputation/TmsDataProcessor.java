package kr.kro.prjectwwtp.imputation;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;

import kr.kro.prjectwwtp.domain.TmsOrigin;
import kr.kro.prjectwwtp.service.TmsOriginService;
import tech.tablesaw.api.*;
import tech.tablesaw.selection.Selection;

public class TmsDataProcessor {
	public Table processTmsData(List<TmsOrigin> originList, LocalDateTime targetDate, ImputationConfig config) {
        // 1. 기초 테이블 생성 (1분 단위 1440개 행 준비)
        Table table = Table.create("TMS_Data");
        DateTimeColumn timeCol = DateTimeColumn.create("tmsTime");
        DoubleColumn tocCol = DoubleColumn.create("toc");
        DoubleColumn phCol = DoubleColumn.create("ph");
        DoubleColumn ssCol = DoubleColumn.create("ss");
        DoubleColumn fluxCol = DoubleColumn.create("flux");
        DoubleColumn tnCol = DoubleColumn.create("tn");
        DoubleColumn tpCol = DoubleColumn.create("tp");

        // 1,440개 타임슬롯 초기화 (결측치 상태)
        LocalDateTime current = targetDate.truncatedTo(ChronoUnit.DAYS);
        for (int i = 0; i < 1440; i++) {
            timeCol.append(current);
            tocCol.append(Double.NaN);
            phCol.append(Double.NaN);
            ssCol.append(Double.NaN);
            fluxCol.append(Double.NaN);
            tnCol.append(Double.NaN);
            tpCol.append(Double.NaN);
            current = current.plusMinutes(1);
        }
        table.addColumns(timeCol, tocCol, phCol, ssCol, fluxCol, tnCol, tpCol);

        // 2. 원본 데이터를 생성된 테이블에 매핑
        for (TmsOrigin origin : originList) {
            int minuteOfDay = origin.getTmsTime().getHour() * 60 + origin.getTmsTime().getMinute();
            if (minuteOfDay < 1440) {
                table.doubleColumn("toc").set(minuteOfDay, origin.getToc());
                table.doubleColumn("ph").set(minuteOfDay, origin.getPh());
                table.doubleColumn("ss").set(minuteOfDay, origin.getSs());
                table.doubleColumn("flux").set(minuteOfDay, origin.getFlux());
                table.doubleColumn("tn").set(minuteOfDay, origin.getTn());
                table.doubleColumn("tp").set(minuteOfDay, origin.getTp());
            }
        }

        // 3. 전략적 보간 수행 (1h = 60min 기준 계산) [cite: 4]
        double freqHours = 1.0 / 60.0; 
        return applyImputationStrategy(table, freqHours, config);
    }

    private Table applyImputationStrategy(Table df, double freqHours, ImputationConfig config) {
        Table dfOut = df.copy();

        for (String colName : new String[]{"toc", "ph", "ss", "flux", "tn", "tp"}) { // 수치형 컬럼 반복 [cite: 5]
            DoubleColumn series = dfOut.doubleColumn(colName);
            
            // Step 1: Forward Fill (단기 결측) [cite: 6]
            int limitShort = (int) (config.getShortTermHours() / freqHours);
            fillForward(series, limitShort);

            // Step 2: EWMA (중기 결측) [cite: 7, 8, 9]
            // Tablesaw에서 EWMA와 Rolling Median은 루프를 통해 구간 길이를 계산하여 적용합니다.
            applyMediumAndLongTerm(series, freqHours, config);
        }
        return dfOut;
    }

    private void fillForward(DoubleColumn col, int limit) {
        int n = col.size();
        
        for (int i = 1; i < n; i++) {
            // 1. 현재 값이 실질적인 결측치(NaN 또는 null)인지 확인
            if (TmsOriginService.isMissing(col, i)) {
                
                // 2. 직전 값(i-1)은 유효한 데이터인지 확인 (채워줄 근거가 있는지)
                if (!TmsOriginService.isMissing(col, i - 1)) {
                    double fillValue = col.get(i - 1);
                    int gapCount = 0;

                    // 3. 결측 구간의 시작점부터 limit 만큼 순회하며 채우기
                    // j < n: 전체 길이를 벗어나지 않음
                    // gapCount < limit: 설정한 시간(3시간 등)만큼만 채움
                    for (int j = i; j < n && gapCount < limit; j++) {
                        if (TmsOriginService.isMissing(col, j)) {
                            col.set(j, fillValue);
                            gapCount++;
                        } else {
                            // 결측 구간이 limit에 도달하기 전에 실제 데이터를 만나면 중단
                            break;
                        }
                    }
                    
                    // 4. 처리한 구간만큼 인덱스 점프 (성능 최적화 및 중복 처리 방지)
                    if (gapCount > 0) {
                        i += (gapCount - 1);
                    }
                }
            }
        }
    }

    // 결측 구간의 길이를 파악하여 중기는 EWMA, 장기는 Rolling Median 적용 [cite: 7, 10]
    // 구체적인 구현은 구간별 인덱스를 추출하여 Tablesaw의 window 기능을 활용합니다.
    private void applyMediumAndLongTerm(DoubleColumn col, double freqHours, ImputationConfig config) {
        int n = col.size();
        int limitShort = (int) (config.getShortTermHours() / freqHours);
        int limitMedium = (int) (config.getMediumTermHours() / freqHours);
        
        // 미리 전체 범위에 대한 보간 컬럼들을 생성 (isMissing 대응 완료된 버전들)
        DoubleColumn ewmaSeries = TmsOriginService.applyEWMA(col, config.getEwmaSpan());
        DoubleColumn rollingMedianSeries = applyRollingMedian(col, (int) (config.getRollingWindow() / freqHours));

        int i = 0;
        while (i < n) {
            // 1. 결측치(NaN/null) 시작 지점 찾기
            if (TmsOriginService.isMissing(col, i)) {
                int start = i;
                // 2. 연속된 결측 구간의 끝 찾기 (isMissing 활용)
                while (i < n && TmsOriginService.isMissing(col, i)) {
                    i++;
                }
                int gapLength = i - start;

                // 3. 구간 길이에 따른 전략 적용
                if (gapLength > limitShort && gapLength <= limitMedium) {
                    // 중기 결측 (예: 4~12시간): EWMA 적용
                    for (int j = start; j < i; j++) {
                        col.set(j, ewmaSeries.get(j));
                    }
                } else if (gapLength > limitMedium) {
                    // 장기 결측 (예: 12시간 이상): Rolling Median 적용
                    for (int j = start; j < i; j++) {
                        col.set(j, rollingMedianSeries.get(j));
                    }
                }
            } else {
                i++;
            }
        }
    }
    
    // Rolling Median 구현 (중앙값 기반 안정적 보간)
    // 파이썬의 series.rolling(window=window, center=True).median() 로직
    private DoubleColumn applyRollingMedian(DoubleColumn col, int windowSize) {
        DoubleColumn result = col.copy();
        int halfWindow = windowSize / 2;
        int n = col.size();

        for (int i = 0; i < n; i++) {
            // 1. 현재 값이 결측치(NaN/Null)인지 통합 체크
            if (TmsOriginService.isMissing(col, i)) {
                int start = Math.max(0, i - halfWindow);
                int end = Math.min(n - 1, i + halfWindow);
                
                // 2. 해당 윈도우 범위의 행 번호(Selection) 추출
                Selection range = Selection.withRange(start, end + 1);
                
                // 3. 해당 범위 내에서 유효한 값(isMissing이 아닌 값)들만 필터링
                // Tablesaw의 where().removeMissing()은 내부적으로 NaN을 걸러냅니다.
                DoubleColumn windowValues = col.where(range).removeMissing();
                
                // 4. 유효한 데이터가 하나라도 있다면 중앙값 계산 후 할당
                if (!windowValues.isEmpty()) {
                    result.set(i, windowValues.median());
                }
            }
        }
        return result;
    }
}
