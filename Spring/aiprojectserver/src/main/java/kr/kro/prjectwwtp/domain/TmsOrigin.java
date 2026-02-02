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
public class TmsOrigin {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long tmsNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime tmsTime;
	@Schema(description = "총유기탄소", example = "double")
	private Double toc;
	@Schema(description = "수소이온농도", example = "double")
	private Double ph;
	@Schema(description = "부유물질", example = "double")
	private Double ss;
	@Schema(description = "유량", example = "int")
	private Integer flux;
	@Schema(description = "총질소", example = "double")
	private Double tn;
	@Schema(description = "총인", example = "double")
	private Double tp;
}