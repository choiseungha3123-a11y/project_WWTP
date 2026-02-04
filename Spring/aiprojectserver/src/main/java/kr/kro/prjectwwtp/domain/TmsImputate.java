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
public class TmsImputate {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@JsonIgnore
	@Schema(description = "고유번호", example = "1~")
	private long tmsNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@JsonProperty("SYS_TIME")
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime tmsTime;
	@JsonProperty("TOC_VU")
	@Schema(description = "총유기탄소", example = "double")
	private Double toc;
	@JsonProperty("PH_VU")
	@Schema(description = "수소이온농도", example = "double")
	private Double ph;
	@JsonProperty("SS_VU")
	@Schema(description = "부유물질", example = "double")
	private Double ss;
	@JsonProperty("FLUX_VU")
	@Schema(description = "유량", example = "int")
	private Integer flux;
	@JsonProperty("TN_VU")
	@Schema(description = "총질소", example = "double")
	private Double tn;
	@JsonProperty("TP_VU")
	@Schema(description = "총인", example = "double")
	private Double tp;
	
	public TmsImputate(TmsOrigin origin) {
		this.tmsTime = origin.getTmsTime();
		this.toc = origin.getToc();
		this.ph = origin.getPh();
		this.ss = origin.getSs();
		this.flux = origin.getFlux();
		this.tn = origin.getTn();
		this.tp = origin.getTp();
	}
}