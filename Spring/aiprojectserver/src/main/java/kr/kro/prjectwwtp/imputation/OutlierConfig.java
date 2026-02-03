package kr.kro.prjectwwtp.imputation;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class OutlierConfig {
    @Builder.Default private String method = "iqr";       // 'iqr' 또는 'zscore' [cite: 12]
    @Builder.Default private double iqrThreshold = 1.5;   // IQR 배수 [cite: 12]
    @Builder.Default private double zscoreThreshold = 3.0;// Z-score 임계값 [cite: 12]
    @Builder.Default private boolean requireBoth = true;  // 도메인+통계 둘 다 해당 시 처리 [cite: 12]
    @Builder.Default private int ewmaSpan = 12;           // EWMA 스팬 [cite: 12]
}
