package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class WeatherApiLog {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long log_no;
	@Schema(description = "로그 종류", example = "Fetch || AddNotEnough || DeleteDuplicate")
	private String logType;
	@Schema(description = "수정 전 개수", example = "0~")
	private int originSize;
	@Schema(description = "수집한 개수", example = "0~")
	private int returnSize;
	@Schema(description = "추가/삭제 한 개수", example = "0~")
	private int modifySize;
	
	@Schema(description = "수집을 위해 요청한 URI", example = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?~~~~")
	private String requestURI;
	@Schema(description = "오류 메시지", example = "처리중 발생한 오류")
	private String errorMsg;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "로그 생성 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime logTime = LocalDateTime.now();
}
