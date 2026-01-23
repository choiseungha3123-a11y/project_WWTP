package kr.kro.prjectwwtp.domain;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class responseDTO {
	private boolean bSuccess;
	private List<Object> dataList;
	private String errorMsg;

	public void addData(Object obj) {
		if(dataList == null)
			dataList = new ArrayList<Object>();
		dataList.add(obj);
	}
}
