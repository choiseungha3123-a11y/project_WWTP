package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.IdClass;
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
@IdClass(WeatherSummaryId.class)
public class WeatherSummary {
	@Id
	@Schema(description = "날짜", example = "2026-01-30")
	private LocalDateTime  time;
	@Id 
	@Schema(description = "측정소 고유번호", example = "368 : 구리 수택동, 569 : 구리 토평동, 541 : 남양주 배양리")
	private int stn;
	@Schema(description = "데이터수", example = "0~")
	private int count;
}
