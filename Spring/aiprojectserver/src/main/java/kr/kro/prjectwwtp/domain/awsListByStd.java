package kr.kro.prjectwwtp.domain;

import java.util.List;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class awsListByStd {
	public List<WeatherDTO> STN_368;
	public List<WeatherDTO> STN_541;
	public List<WeatherDTO> STN_569;
	public awsListByStd(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569) {
		this.STN_368 = aws368;
		this.STN_541 = aws541;
		this.STN_569 = aws569;
	}
}