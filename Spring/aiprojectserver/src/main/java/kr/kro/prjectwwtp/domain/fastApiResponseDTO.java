package kr.kro.prjectwwtp.domain;

import java.util.Dictionary;

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
public class fastApiResponseDTO {
	@Schema(description = "고유값")
	private String request_id;
	@Schema(description = "성공실패 유부", example = "true | false")
	private boolean ok;
	@Schema(description = "예측치 Dictonary")
	private Dictionary<String, Object> output;
	@Schema(description = "예측 소요시간(ms)", example = "731")
	private int latency_ms;
	@Schema(description = "오류값")
	private String error;
}

