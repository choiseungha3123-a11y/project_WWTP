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
public class Weather {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long dataNo; // 필드명을 CamelCase로 변경
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	@Builder.Default
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime logTime = LocalDateTime.now();
	@Schema(description = "측정소 고유번호", example = "368 : 구리 수택동, 569 : 구리 토평동, 541 : 남양주 배양리")
	private int stn;
	
	@Schema(description = "1분 평균 풍향 (degree) : 0-N, 90-E, 180-S, 270-W, 360-무풍")
	private double wd1;
	@Schema(description = "1분 평균 풍속 (m/s)")
	private double wd2;
	@Schema(description = "최대 순간 풍향 (degree)")
	private double wds;
	@Schema(description = "최대 순간 풍속 (m/s)")
	private double wss;
	@Schema(description = "10분 평균 풍향 (degree)")
	private double wd10;
	@Schema(description = "10분 평균 풍속 (m/s)")
	private double ws10;
	@Schema(description = "1분 평균 기온 (C)")
	private double ta;
	@Schema(description = "강수감지 (0-무강수, 0이 아니면-강수)")
	private double re;
	@Schema(description = "15분 누적 강수량 (mm)")
	private double rn15m;
	@Schema(description = "60분 누적 강수량 (mm)")
	private double rn60m;
	@Schema(description = "12시간 누적 강수량 (mm)")
	private double rn12h;
	@Schema(description = "일 누적 강수량 (mm)")
	private double rnday;
	@Schema(description = "1분 평균 상대습도 (%)")
	private double hm;
	@Schema(description = "1분 평균 현지기압 (hPa)")
	private double pa;
	@Schema(description = "1분 평균 해면기압 (hPa)")
	private double ps;
	@Schema(description = "이슬점온도 (C)")
	private double td; 
}
