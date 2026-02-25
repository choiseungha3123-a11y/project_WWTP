package kr.kro.prjectwwtp.domain;

import java.util.List;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class Input<T> {
	public awsListByStd awsList;
	public List<T> dataList;
	public Input(List<WeatherDTO> aws368, List<WeatherDTO> aws541, List<WeatherDTO> aws569, List<T> flowList) {
		this.awsList = new awsListByStd( aws368, aws541, aws569);
		this.dataList = flowList;
	}
}
