package kr.kro.prjectwwtp.domain;

import java.time.format.DateTimeFormatter;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class WeatherDTO {
	@JsonIgnore
	@Schema(description = "고유번호", example = "1~")
	long dataNo;
	@JsonProperty("SYS_TIME")
	@Schema(description = "데이터의 기록 시간", example = "2026-01-30T15:30:00")
	String time;
//	@JsonProperty("STN")
//	@Schema(description = "측정소 고유번호", example = "368 : 구리 수택동, 569 : 구리 토평동, 541 : 남양주 배양리")
//	int stn;
	@JsonProperty("TA")
	@Schema(description = "1분 평균 기온 (C)")
	Double ta;
	@JsonProperty("RN_15m")
	@Schema(description = "15분 누적 강수량 (mm)")
	Double rn15m;
	@JsonProperty("RN_60m")
	@Schema(description = "60분 누적 강수량 (mm)")
	Double rn60m;
	@JsonProperty("RN_12H")
	@Schema(description = "12시간 누적 강수량 (mm)")
	Double rn12h;
	@JsonProperty("RN_DAY")
	@Schema(description = "일 누적 강수량 (mm)")
	Double rnday;
	@JsonProperty("HM")
	@Schema(description = "1분 평균 상대습도 (%)")
	Double hm;
	@JsonProperty("TD")
	@Schema(description = "이슬점온도 (C)")
	Double td; 
	@JsonProperty("distance")
	@Schema(description = "처리장과의 거리")
	double distance;
	
	public WeatherDTO(Weather data) {
		this.dataNo = data.getDataNo();
		this.time = data.getLogTime().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
//		this.stn = data.getStn();
		this.ta = data.getTa();
		this.rn15m = data.getRn15m();
		this.rn60m = data.getRn60m();
		this.rn12h = data.getRn12h();
		this.rnday = data.getRnday();
		this.hm = data.getHm();
		this.td = data.getTd();
		switch(data.getStn()) {
			case 368:
				this.distance = 1.02f;
				break;
			case 541:
				this.distance = 4.61f;
				break;
			case 569:
				this.distance = 1.24f;
				break;
		}
	}
}
