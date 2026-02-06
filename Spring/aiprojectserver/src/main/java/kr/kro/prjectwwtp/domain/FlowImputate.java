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
public class FlowImputate {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@JsonIgnore
	@Schema(description = "고유번호", example = "1~")
	private long flowNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@JsonProperty("SYS_TIME")
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime flowTime;
	@JsonProperty("flow_TankA")
	@Schema(description = "유량조정조A_유량", example = "double")
	private Double flowA;
	@JsonProperty("flow_TankB")
	@Schema(description = "유량조정조B_유량", example = "double")
	private Double flowB;
	@JsonProperty("level_TankA")
	@Schema(description = "유량조정조A_수위", example = "double")
	private Double levelA;
	@JsonProperty("level_TankB")
	@Schema(description = "유량조정조B_수위", example = "double")
	private Double levelB;
	@JsonProperty("Q_in")
	@Schema(description = "유량조정조 유량 A+B", example = "double")
	private Double sum;
	
	public FlowImputate(FlowOrigin origin) {
		this.flowTime = origin.getFlowTime();
		this.flowA = origin.getFlowA();
		this.flowB = origin.getFlowB();
		this.levelA = origin.getLevelA();
		this.levelB = origin.getLevelB();
		this.sum = this.flowA + flowB;
	}
}