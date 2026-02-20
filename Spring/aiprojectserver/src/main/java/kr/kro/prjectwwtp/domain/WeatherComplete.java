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
public class WeatherComplete {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "data_no") // DB 컬럼명은 그대로 유지
	@Schema(description = "고유번호", example = "1~")
	private long dataNo; // 필드명을 CamelCase로 변경
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Schema(description = "데이터 날짜", example = "2026-01-30T15:30:00")
	private LocalDateTime dataTime;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "생성 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime createTime = LocalDateTime.now();
	@Schema(description = "측정소 고유번호", example = "368 : 구리 수택동, 569 : 구리 토평동, 541 : 남양주 배양리")
	int stn;
	@Schema(description = "데이터 크기", example = "24 * 60 = 1440")
	int dataSize;
}
