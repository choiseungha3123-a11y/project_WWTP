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
public class FlowOrigin {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long flowNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime flowTime;
	@Schema(description = "유량조정조A_유량", example = "double")
	private Double flowA;
	@Schema(description = "유량조정조B_유량", example = "double")
	private Double flowB;
	@Schema(description = "유량조정조A_수위", example = "double")
	private Double levelA;
	@Schema(description = "유량조정조B_수위", example = "double")
	private Double levelB;
}
