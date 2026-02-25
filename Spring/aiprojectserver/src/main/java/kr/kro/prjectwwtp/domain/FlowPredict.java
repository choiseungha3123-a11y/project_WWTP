package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

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
public class FlowPredict {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@JsonIgnore
	@Schema(description = "고유번호", example = "1~")
	private long flowNo;
	@JsonProperty("SYS_TIME")
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@Schema(description = "데이터의 예측 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime flowTime;
	@JsonProperty("Q_in")
	@Schema(description = "예측_유량", example = "double")
	private Double flowValue;
	@JsonIgnore
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "createtime", updatable = false)
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	@Builder.Default
	private LocalDateTime createTime = LocalDateTime.now();
}
