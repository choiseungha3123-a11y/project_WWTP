package kr.kro.prjectwwtp.imputation;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ImputationConfig {
    @Builder.Default private int shortTermHours = 3;      // 단기 결측: 1-3시간 [cite: 1]
    @Builder.Default private int mediumTermHours = 12;    // 중기 결측: 4-12시간 [cite: 1]
    @Builder.Default private int ewmaSpan = 6;            // EWMA 스팬 [cite: 1]
    @Builder.Default private int rollingWindow = 24;      // Rolling median 윈도우 [cite: 1]
}