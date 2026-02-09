package kr.kro.prjectwwtp.domain;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class predictIn<T> {
	private String predict_Id;
	private Input<T> in;
	public predictIn(Input<T> in) {
		this.in = in;
	}
}
