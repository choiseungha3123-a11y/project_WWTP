package kr.kro.prjectwwtp.domain;

import java.util.ArrayList;
import java.util.List;

import io.swagger.v3.oas.annotations.media.ArraySchema;
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
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Schema(description = "API 응답")
public class responseDTO {
	@Schema(description = "성공/실패", example = "true")
	private boolean success;
	@Schema(description = "dataList의 데이터 수 : 0~", example = "0")
	private int dataSize;
	@ArraySchema(schema = @Schema(description = "전달할 데이터 : null or []", example = "null || []"))
	private List<Object> dataList;
	@Schema(description = "success 가 false 일때의 원인", example = "정보가 올바르지 않습니다.")
	private String errorMsg;

	public void addData(Object obj) {
		if(dataList == null) {
			dataSize = 0;
			dataList = new ArrayList<Object>();
		}
		dataList.add(obj);
		++dataSize;
	}
}
